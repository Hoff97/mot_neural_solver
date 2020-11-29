from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
import torch
from collections import OrderedDict
from torch.jit.annotations import Tuple, List
from torchvision.models.detection.roi_heads import keypointrcnn_inference
from torchvision.models.detection.transform import resize_keypoints

class KeypointOnlyRCNN(KeypointRCNN):
    def forward(self, images, bounding_boxes):
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, bounding_boxes = self.transform(images, [{"boxes": bounding_boxes[0]}])
        bounding_boxes = [bounding_boxes[0]["boxes"]]
        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        image_shapes = images.image_sizes

        keypoints_probs, kp_scores = self.get_joints(features, bounding_boxes, image_shapes)

        keypoints = resize_keypoints(keypoints_probs[0], image_shapes[0], original_image_sizes[0])

        return keypoints

    def get_joints(self, features, keypoint_proposals, image_shapes):
        keypoint_features = self.roi_heads.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
        keypoint_features = self.roi_heads.keypoint_head(keypoint_features)
        keypoint_logits = self.roi_heads.keypoint_predictor(keypoint_features)

        assert keypoint_logits is not None
        assert keypoint_proposals is not None

        keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
        return keypoints_probs, kp_scores


model_urls = {
    # legacy model for BC reasons, see https://github.com/pytorch/vision/issues/1606
    'keypointrcnn_resnet50_fpn_coco_legacy':
        'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth',
    'keypointrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
}


def keypointonlyrcnn_resnet50_fpn(pretrained=False, progress=True,
                                  num_classes=2, num_keypoints=17,
                                  pretrained_backbone=True, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = KeypointOnlyRCNN(backbone, num_classes, num_keypoints=num_keypoints, **kwargs)
    if pretrained:
        key = 'keypointrcnn_resnet50_fpn_coco'
        if pretrained == 'legacy':
            key += '_legacy'
        state_dict = load_state_dict_from_url(model_urls[key],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model