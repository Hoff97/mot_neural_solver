import os
import os.path as osp

import pandas as pd
import pytorch_lightning as pl
import torch
from mot_neural_solver.data.mot_graph_dataset import MOTGraphDataset
from mot_neural_solver.models.combined.mp import MOTMPNet as CombinedMOTMPNet
from mot_neural_solver.models.mpn import MOTMPNet
from mot_neural_solver.models.resnet import load_pretrained_weights, resnet50_fc256
from mot_neural_solver.path_cfg import OUTPUT_PATH
from mot_neural_solver.tracker.mpn_tracker import MPNTracker
from mot_neural_solver.utils.evaluation import compute_perform_metrics
from torch import optim as optim_module
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler as lr_sched_module
from torch_geometric.data import DataLoader


class MOTNeuralSolver(pl.LightningModule):
    """
    Pytorch Lightning wrapper around the MPN defined in model/mpn.py.
    (see https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html)

    It includes all data loading and train / val logic., and it is used for both training and testing models.
    """

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.model, self.cnn_model = self.load_model()

    def forward(self, x):
        self.model(x)

    def load_model(self):
        cnn_arch = self.hparams["graph_model_params"]["cnn_params"]["arch"]

        if (
            "multi" not in self.hparams["graph_model_params"]
            or not self.hparams["graph_model_params"]["multi"]
        ):
            model = MOTMPNet(self.hparams["graph_model_params"]).cuda()
        else:
            model = CombinedMOTMPNet(self.hparams["graph_model_params"]).cuda()

        cnn_model = resnet50_fc256(10, loss="xent", pretrained=True).cuda()
        load_pretrained_weights(
            cnn_model,
            osp.join(
                OUTPUT_PATH,
                self.hparams["graph_model_params"]["cnn_params"]["model_weights_path"][
                    cnn_arch
                ],
            ),
        )
        cnn_model.return_embeddings = True

        return model, cnn_model

    def _get_data(self, mode, return_data_loader=True):
        assert mode in ("train", "val", "test")

        dataset = MOTGraphDataset(
            dataset_params=self.hparams["dataset_params"],
            mode=mode,
            cnn_model=self.cnn_model,
            splits=self.hparams["data_splits"][mode],
            logger=None,
        )

        if return_data_loader and len(dataset) > 0:
            train_dataloader = DataLoader(
                dataset,
                batch_size=self.hparams["train_params"]["batch_size"],
                shuffle=True if mode == "train" else False,
                num_workers=self.hparams["train_params"]["num_workers"],
            )
            return train_dataloader

        elif return_data_loader and len(dataset) == 0:
            return []

        else:
            return dataset

    def train_dataloader(self):
        return self._get_data(mode="train")

    def val_dataloader(self):
        return self._get_data("val")

    def test_dataset(self, return_data_loader=False):
        return self._get_data("test", return_data_loader=return_data_loader)

    def configure_optimizers(self):
        optim_class = getattr(
            optim_module, self.hparams["train_params"]["optimizer"]["type"]
        )
        optimizer = optim_class(
            self.model.parameters(), **self.hparams["train_params"]["optimizer"]["args"]
        )

        if self.hparams["train_params"]["lr_scheduler"]["type"] is not None:
            lr_sched_class = getattr(
                lr_sched_module, self.hparams["train_params"]["lr_scheduler"]["type"]
            )
            lr_scheduler = lr_sched_class(
                optimizer, **self.hparams["train_params"]["lr_scheduler"]["args"]
            )

            return [optimizer], [lr_scheduler]

        else:
            return optimizer

    def _compute_loss(self, outputs, batch):
        if (
            self.hparams["train_params"]["loss"] is None
            or self.hparams["train_params"]["loss"]["type"] == "BCE"
        ):
            # Define Balancing weight
            positive_vals = batch.edge_labels.sum()

            if positive_vals:
                pos_weight = (
                    batch.edge_labels.shape[0] - positive_vals
                ) / positive_vals

            else:  # If there are no positives labels, avoid dividing by zero
                pos_weight = 0

            # Compute Weighted BCE:
            loss = 0
            num_steps = len(outputs["classified_edges"])
            for step in range(num_steps):
                output = outputs["classified_edges"][step]
                if (
                    "multi" in self.hparams["graph_model_params"]
                    and self.hparams["graph_model_params"]["multi"]
                ):
                    output = output["bb-bb"]

                loss += F.binary_cross_entropy_with_logits(
                    output.view(-1),
                    batch.edge_labels.view(-1),
                    pos_weight=pos_weight,
                )

            return loss
        elif self.hparams["train_params"]["loss"]["type"] == "Focal":
            alpha = self.hparams["train_params"]["loss"]["args"]["alpha"]

            loss = 0
            num_steps = len(outputs["classified_edges"])
            for step in range(num_steps):
                x = outputs["classified_edges"][step].view(-1)
                t = Variable(batch.edge_labels.view(-1)).cuda()

                xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
                pt = (2 * xt + 1).sigmoid()

                w = alpha * t + (1 - alpha) * (1 - t)
                loss += (-w * pt.log() / 2).sum()

            return loss
        else:
            raise Exception(f'Loss {self.hparams["train_params"]["loss"]} unknown')

    def _train_val_step(self, batch, batch_idx, train_val):
        device = (next(self.model.parameters())).device
        batch.to(device)

        outputs = self.model(batch)
        loss = self._compute_loss(outputs, batch)
        logs = {**compute_perform_metrics(outputs, batch), **{"loss": loss}}
        log = {key + f"/{train_val}": val for key, val in logs.items()}

        if train_val == "train":
            return {"loss": loss, "log": log}

        else:
            return log

    def training_step(self, batch, batch_idx):
        return self._train_val_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._train_val_step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs):
        metrics = pd.DataFrame(outputs).mean(axis=0).to_dict()
        metrics = {
            metric_name: torch.as_tensor(metric)
            for metric_name, metric in metrics.items()
        }
        return {"val_loss": metrics["loss/val"], "log": metrics}

    def track_all_seqs(self, output_files_dir, dataset, use_gt=False, verbose=False):
        tracker = MPNTracker(
            dataset=dataset,
            graph_model=self.model,
            use_gt=use_gt,
            eval_params=self.hparams["eval_params"],
            dataset_params=self.hparams["dataset_params"],
        )

        constraint_sr = pd.Series(dtype=float)
        for seq_name in dataset.seq_names:
            print("Tracking", seq_name)
            if verbose:
                print("Tracking sequence ", seq_name)

            os.makedirs(output_files_dir, exist_ok=True)
            _, constraint_sr[seq_name] = tracker.track(
                seq_name, output_path=osp.join(output_files_dir, seq_name + ".txt")
            )

            if verbose:
                print("Done! \n")

        constraint_sr["OVERALL"] = constraint_sr.mean()

        return constraint_sr
