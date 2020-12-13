import mot_neural_solver.models.mmpose_utils as utils

"""
This is a wrapper file for the inclusion of mmpose (https://github.com/open-mmlab/mmpose) top-down models as joint-detectors.
"""

MMPOSE_CFGS_PATH = '../mmpose/configs'
MMPOSE_MODEL_INFO = dict(
    mmpose_pose_resnet101=dict(
        ckp_link='https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth',
        cfg_file=f'{MMPOSE_CFGS_PATH}/top_down/resnet/coco/res101_coco_256x192.py',
        in_size=(256, 192)
    ),
    mmpose_pose_resnet152=dict(
        ckp_link='https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288-3860d4c9_20200709.pth',
        cfg_file=f'{MMPOSE_CFGS_PATH}/top_down/resnet/coco/res152_coco_256x192.py',
        in_size=(256, 192)
    ),
    mmpose_dark_pose_hrnet_w48=dict(
        ckp_link='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_dark-8cba3197_20200812.pth',
        cfg_file=f'{MMPOSE_CFGS_PATH}/top_down/darkpose/coco/hrnet_w48_coco_256x192_dark.py',
        in_size=(256, 192)
    ),
    mmpose_dark_pose_hrnet_w32=dict(
        ckp_link='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_dark-07f147eb_20200812.pth',
        cfg_file=f'{MMPOSE_CFGS_PATH}/top_down/darkpose/coco/hrnet_w32_coco_256x192_dark.py',
        in_size=(256, 192)
    ),
    mmpose_hrnet_w32=dict(
        ckp_link='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        cfg_file=f'{MMPOSE_CFGS_PATH}/top_down/hrnet/coco/hrnet_w32_coco_256x192.py',
        in_size=(256, 192)
    ),
    mmpose_hrnet_w48=dict(
        ckp_link='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
        cfg_file=f'{MMPOSE_CFGS_PATH}/top_down/hrnet/coco/hrnet_w48_coco_256x192.py',
        in_size=(256, 192)
    ),
)


def top_down_model(model_name):
    """
    Loads a supported top-down model from MMPOSE pre-trained on COCO challenge.
    """
    assert model_name in MMPOSE_MODEL_INFO.keys(), f"mmpose error: model '{model_name}' not supported."

    model_info = MMPOSE_MODEL_INFO[model_name]
    cfg = utils.load_mmpose_config(model_info['cfg_file'])
    model = utils.load_model(cfg, model_info['ckp_link'])

    return model


def model_input_size(model_name):
    """
    Returns mmpose model's input size.
    """
    return MMPOSE_MODEL_INFO[model_name]['in_size']
