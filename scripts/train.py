import itertools
import os
import os.path as osp
import yaml

import sacred
from sacred import SETTINGS, Experiment
from mot_neural_solver.path_cfg import OUTPUT_PATH
from mot_neural_solver.pl_module.pl_module import MOTNeuralSolver
from mot_neural_solver.utils.evaluation import MOTMetricsLogger
from mot_neural_solver.utils.misc import (
    ModelCheckpoint,
    get_run_str_and_save_dir,
    make_deterministic,
)
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger

# from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
CONFIG_FILE = "configs/tracking_cfg_combined.yaml"

ex = Experiment()
ex.add_config(CONFIG_FILE)
ex.add_config(
    {"run_id": "train_w_default_config", "add_date": True, "cross_val_split": None}
)

@ex.config
def cfg(cross_val_split, eval_params, dataset_params, graph_model_params, data_splits):

    # Training requires the use of precomputed embeddings
    assert dataset_params[
        "precomputed_embeddings"
    ], "Training without precomp. embeddings is not supp"

    assert dataset_params[
        "precomputed_joints"
    ], "Training without precomp. joint detections is not supported"

    # Only use tracktor for postprocessing if tracktor was used for preprocessing
    if "tracktor" not in dataset_params["det_file_name"]:
        eval_params["add_tracktor_detects"] = False

    # Determine which sequences will be used for training  / validation
    if cross_val_split is not None:
        assert cross_val_split in (
            1,
            2,
            3,
        ), f"{cross_val_split} is not a valid cross validation split"
        data_splits["train"] = [
            "mot15_train_gt",
            f"mot17_split_{cross_val_split}_train_gt",
        ]
        data_splits["val"] = [f"split_{cross_val_split}_val"]

    # If we're training on all the available training data, disable validation
    if data_splits["train"] == "all_train" or data_splits["val"] is None:
        data_splits["val"] = []
        eval_params["val_percent_check"] = 0

@ex.main
def main(_config, _run):
    sacred.commands.print_config(_run)
    seed_everything(12345)

    model = MOTNeuralSolver(hparams=dict(_config))

    run_str, save_dir = get_run_str_and_save_dir(
        _config["run_id"], _config["cross_val_split"], _config["add_date"]
    )

    if _config["train_params"]["tensorboard"]:
        logger = TensorBoardLogger(OUTPUT_PATH, name="experiments", version=run_str)

    else:
        logger = None

    ckpt_callback = ModelCheckpoint(
        save_epoch_start=_config["train_params"]["save_epoch_start"],
        save_every_epoch=_config["train_params"]["save_every_epoch"],
    )
    mot_callback = MOTMetricsLogger(
        compute_oracle_results=_config["eval_params"]["normalize_mot_metrics"]
    )

    callbacks = [ckpt_callback, mot_callback]

    if _config["train_params"]["lr_scheduler"]["type"] is not None:
        lr_logger = LearningRateLogger()
        callbacks.append(lr_logger)

    trainer = Trainer(
        gpus=1,
        callbacks=callbacks,
        weights_summary=None,
        checkpoint_callback=False,
        max_epochs=_config["train_params"]["num_epochs"],
        val_percent_check=_config["eval_params"]["val_percent_check"],
        check_val_every_n_epoch=_config["eval_params"]["check_val_every_n_epoch"],
        nb_sanity_val_steps=0,
        logger=logger,
        default_save_path=osp.join(OUTPUT_PATH, "experiments", run_str),
        deterministic=True,
    )
    trainer.fit(model)

def grid_search(conf):
    """
    Performs grid search on some of the hyperparameters, as specified in tracking_cfg.yaml.
    """
    grid_search = conf["grid_search"]
    ranges = []
    for p, l in grid_search.items():
        ranges.append(range(len(l)))

    # get all possible combinations of params
    combinations = output_list = list(itertools.product(*ranges))

    for comb in combinations:
        params = list(grid_search.keys())
        param_pairs = {}
        for i in range(len(comb)):
            param_pairs[params[i]] = grid_search[params[i]][comb[i]]
        yield param_pairs

# dirty hack to allow grid_search to be specified in same config file as the experiment
def read_config(config_file):
    with open(config_file, "r") as f:
            return yaml.safe_load(f)

def updatedict(d, key, val):
    """
    Recursively iterates through a dict and updates a given key's value.
    """
    new_d = {}
    for k, v in d.items():
        if key == k:
            new_d[key] = val
        elif isinstance(v, dict):
            new_d[k] = updatedict(v, key, val)
        else:
            new_d[k] = v

    return new_d

def run(ex):
    """
    Runs experiment with regard to run seperate experiments for parameter searches.
    """
    conf = read_config(CONFIG_FILE)
    # check if parameter search is wanted
    param_search = conf["parameter_search"]

    if param_search is False:
        ex.run()
    else:
        assert param_search.lower() in ["grid_search"], f"Unsupported parameter search: '{param_search}'"

        new_conf = conf
        if param_search == "grid_search":
            for new_params in grid_search(conf):
                # update params in config
                for k, v in new_params.items():
                    print(f"RUNNING WITH --> {k}: {v}")
                    new_conf = updatedict(new_conf, k, v)

                # run experiment with updated config from the parameter search
                run = ex.run(config_updates=new_conf)

if __name__ == "__main__":
    run(ex)
