import mot_neural_solver.models.mmpose_utils as utils

"""
This is a wrapper file for the inclusion of mmpose (https://github.com/open-mmlab/mmpose) top-down models as joint-detectors.
"""

MMPOSE_CFGS_PATH = '../mmpose/'
MMPOSE_MODEL_INFO = dict(
    mmpose_keypoint_resnet101=dict(
        ckp_link='https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth',
        cfg_file=f'{MMPOSE_CFGS_PATH}configs/top_down/resnet/coco/res101_coco_256x192.py',
        in_size=(256, 192)
    )
)

def top_down_model(model_name):
    """
    Loads a supported top-down model from MMPOSE pre-trained on COCO challenge.
    """
    if model_name == 'mmpose_keypoint_resnet101':
        model_info = MMPOSE_MODEL_INFO[model_name]
        cfg = utils.load_mmpose_config(model_info['cfg_file'])
        model = utils.load_model(cfg, model_info['ckp_link'])
    # TODO: add more models
    else:
        sys.exit(f'error: model {model_name} not supported.')

    return model


def model_input_size(model_name):
    """
    Returns mmpose model's input size.
    """
    return MMPOSE_MODEL_INFO[model_name]['in_size']
