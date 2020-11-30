import os.path as osp

import numpy as np

from PIL import Image
from skimage.io import imread
#from skimage.util import pad
from numpy import pad

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from torchvision.models.detection import keypointrcnn_resnet50_fpn
from mot_neural_solver.models.keypointonly_rcnn import keypointonlyrcnn_resnet50_fpn

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

class BoundingBoxDataset(Dataset):
    """
    Class used to process detections. Given a DataFrame (det_df) with detections of a MOT sequence, it returns
    the image patch corresponding to the detection's bounding box coordinates
    """
    def __init__(self, det_df, seq_info_dict, pad_ = True, pad_mode = 'mean', output_size = (128, 64),
                 return_det_ids_and_frame = False):
        self.det_df = det_df
        self.seq_info_dict = seq_info_dict
        self.pad = pad_
        self.pad_mode = pad_mode
        self.transforms= Compose((Resize(output_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])))

        # Initialize two variables containing the path and img of the frame that is being loaded to avoid loading multiple
        # times for boxes in the same image
        self.curr_img = None
        self.curr_img_path = None

        self.return_det_ids_and_frame = return_det_ids_and_frame

    def __len__(self):
        return self.det_df.shape[0]

    def __getitem__(self, ix):
        row = self.det_df.iloc[ix]

        # Load this bounding box' frame img, in case we haven't done it yet
        if row['frame_path'] != self.curr_img_path:
            self.curr_img = imread(row['frame_path'])
            self.curr_img_path = row['frame_path']

        frame_img = self.curr_img

        # Crop the bounding box, and pad it if necessary to
        bb_img = frame_img[int(max(0, row['bb_top'])): int(max(0, row['bb_bot'])),
                   int(max(0, row['bb_left'])): int(max(0, row['bb_right']))]
        if self.pad:
            x_height_pad = np.abs(row['bb_top'] - max(row['bb_top'], 0)).astype(int)
            y_height_pad = np.abs(row['bb_bot'] - min(row['bb_bot'], self.seq_info_dict['frame_height'])).astype(int)

            x_width_pad = np.abs(row['bb_left'] - max(row['bb_left'], 0)).astype(int)
            y_width_pad = np.abs(row['bb_right'] - min(row['bb_right'], self.seq_info_dict['frame_width'])).astype(int)

            bb_img = pad(bb_img, ((x_height_pad, y_height_pad), (x_width_pad, y_width_pad), (0, 0)), mode=self.pad_mode)

        bb_img = Image.fromarray(bb_img)
        if self.transforms is not None:
            bb_img = self.transforms(bb_img)

        if self.return_det_ids_and_frame:
            return row['frame'], row['detection_id'], bb_img
        else:
            return bb_img

class FrameDataset(Dataset):
    """
    Class used to get image frames d
    """
    def __init__(self, det_df, seq_info_dict):
        self.det_df = det_df
        self.seq_info_dict = seq_info_dict
        self.transforms=Compose([ToTensor()])

        # Initialize two variables containing the path and img of the frame that is being loaded to avoid loading multiple
        # times for boxes in the same image
        self.curr_img = None
        self.curr_img_path = None

        self.frames = self.det_df.frame_path.unique()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, ix):
        """Gets a frame and the bounding boxes
        The BB-Boxes is returned as a np array with shape [N,4]
        where the columns are x1, y1, x2, y2
        """
        frame_path = self.frames[ix]

        # Load this bounding box' frame img, in case we haven't done it yet
        if frame_path != self.curr_img_path:
            self.curr_img = imread(frame_path)
            self.curr_img_path = frame_path

        frame_img = self.curr_img
        frame_img = Image.fromarray(frame_img)
        if self.transforms is not None:
            frame_img = self.transforms(frame_img)

        detections = self.det_df.loc[self.det_df['frame_path'] == frame_path]

        bounding_boxes = np.zeros((len(detections),4))

        bounding_boxes[:, 0] = detections.bb_left
        bounding_boxes[:, 1] = detections.bb_top
        bounding_boxes[:, 2] = detections.bb_right
        bounding_boxes[:, 3] = detections.bb_bot

        ids = np.array(detections.id.array)

        return frame_img, bounding_boxes, frame_path, detections.frame.unique()[0], ids

def load_embeddings_from_imgs(det_df, dataset_params, seq_info_dict, cnn_model, return_imgs = False, use_cuda=True):
    """
    Computes embeddings for each detection in det_df with a CNN.
    Args:
        det_df: pd.DataFrame with detection coordinates
        seq_info_dict: dict with sequence meta info (we need frame dims)
        cnn_model: CNN to compute embeddings with. It needs to return BOTH node embeddings and reid embeddings
        return_imgs: bool, determines whether RGB images must also be returned

    Returns:
        (bb_imgs for each det or [], torch.Tensor with shape (num_detects, node_embeddings_dim), torch.Tensor with shape (num_detects, reidembeddings_dim))

    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    ds = BoundingBoxDataset(det_df, seq_info_dict = seq_info_dict, output_size=dataset_params['img_size'],
                            return_det_ids_and_frame=False)
    bb_loader = DataLoader(ds, batch_size=dataset_params['img_batch_size'], pin_memory=True, num_workers=6)
    cnn_model = cnn_model.eval()

    bb_imgs = []
    node_embeds = []
    reid_embeds = []
    with torch.no_grad():
        for bboxes in bb_loader:
            node_out, reid_out = cnn_model(bboxes.cuda())
            node_embeds.append(node_out.to(device))
            reid_embeds.append(reid_out.to(device))

            if return_imgs:
                bb_imgs.append(bboxes)

    node_embeds = torch.cat(node_embeds, dim=0)
    reid_embeds = torch.cat(reid_embeds, dim=0)

    return bb_imgs, node_embeds, reid_embeds

def load_joints_from_imgs(det_df, dataset_params, seq_info_dict, use_cuda=True):
    """
    Computes embeddings for each detection in det_df with a CNN.
    Args:
        det_df: pd.DataFrame with detection coordinates
        seq_info_dict: dict with sequence meta info (we need frame dims)

    Returns:
        (bb_imgs for each det or [], torch.Tensor with shape (num_detects, node_embeddings_dim), torch.Tensor with shape (num_detects, reidembeddings_dim))

    """

    ds = FrameDataset(det_df, seq_info_dict = seq_info_dict)
    loader = DataLoader(ds, batch_size=1, pin_memory=True, num_workers=0)

    model = keypointonlyrcnn_resnet50_fpn(pretrained=True).eval().cuda()

    with torch.no_grad():
        for frame, bb_boxes, frame_path, _, ids in tqdm(loader):
            # This assumes that joint_img_batch_size is 1
            keypoints, kp_scores = model([frame[0].cuda()], [bb_boxes[0].float().cuda()])
            # TODO: Aggregate and return detected joints

    return None

def plot_img_with_bb(img: np.ndarray, bb_boxes: np.ndarray,
                     keypoints: np.ndarray, ids:np.ndarray, save_path: str):
    img = (img*255).astype(np.uint8)
    img = img.transpose((1,2,0))

    fig,ax = plt.subplots(1)
    ax.imshow(img)

    colors=['b','g','r','c','m','y']

    for i in range(bb_boxes.shape[0]):
        bb_box = bb_boxes[i]
        color = colors[ids[i] % len(colors)]
        # Create a Rectangle patch
        rect = patches.Rectangle((bb_box[0], bb_box[1]),bb_box[2]-bb_box[0],bb_box[3]-bb_box[1],linewidth=1,edgecolor=color,facecolor='none')
        ax.add_patch(rect)

        kps = keypoints[i]
        for j in range(kps.shape[0]):
            kp = kps[j]
            circle = plt.Circle((kp[0], kp[1]), 5, color=color)
            ax.add_artist(circle)

    plt.savefig(save_path)
    plt.close('all')

def load_precomputed_embeddings(det_df, seq_info_dict, embeddings_dir, use_cuda):
    """
    Given a sequence's detections, it loads from disk embeddings that have already been computed and stored for its
    detections
    Args:
        det_df: pd.DataFrame with detection coordinates
        seq_info_dict: dict with sequence meta info (we need frame dims)
        embeddings_dir: name of the directory where embeddings are stored

    Returns:
        torch.Tensor with shape (num_detects, embeddings_dim)

    """
    # Retrieve the embeddings we need from their corresponding locations
    embeddings_path = osp.join(seq_info_dict['seq_path'], 'processed_data', 'embeddings', seq_info_dict['det_file_name'],
                               embeddings_dir)
    #print("EMBEDDINGS PATH IS ", embeddings_path)
    frames_to_retrieve = sorted(det_df.frame.unique())
    embeddings_list = [torch.load(osp.join(embeddings_path, f"{frame_num}.pt")) for frame_num in frames_to_retrieve]
    embeddings = torch.cat(embeddings_list, dim=0)

    # First column in embeddings is the index. Drop the rows of those that are not present in det_df
    ixs_to_drop = list(set(embeddings[:, 0].int().numpy()) - set(det_df['detection_id']))
    embeddings = embeddings[~np.isin(embeddings[:, 0], ixs_to_drop)]  # Not so clean, but faster than a join
    assert_str = "Problems loading embeddings. Indices between query and stored embeddings do not match. BOTH SHOULD BE SORTED!"
    assert (embeddings[:, 0].numpy() == det_df['detection_id'].values).all(), assert_str

    embeddings = embeddings[:, 1:]  # Get rid of the detection index

    return embeddings.to(torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"))

def load_precomputed_joints(det_df, seq_info_dict, joints_dir, use_cuda):
    joints_path = osp.join(seq_info_dict['seq_path'], 'processed_data', 'joints', seq_info_dict['det_file_name'],
                               joints_dir)
    frames = sorted(det_df.frame.unique())
    joints = [torch.load(osp.join(joints_path, f"{frame}.pt")) for frame in frames]
    joints = torch.cat(joints, dim=0)

    # ixs_to_drop = list(set(embeddings[:, 0].int().numpy()) - set(det_df['detection_id']))
    # TODO: Drop joints of BB not present in det_df?
    # TODO: There are probably multiple frames in det_df, so we have to load all of them

    return joints.to(torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"))