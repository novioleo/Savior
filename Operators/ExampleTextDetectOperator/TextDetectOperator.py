from abc import ABC

import numpy as np

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Operators.ExampleTextDetectOperator.PostProcess import db_post_process
from Utils.GeometryUtils import resize_with_short_side, resize_with_specific_base, force_convert_image_to_bgr, \
    resize_with_long_side, center_pad_image_with_specific_base
from Utils.InferenceHelpers import TritonInferenceHelper


class TextDetectOperator(DummyAlgorithmWithModel, ABC):
    name = 'TextDetect'
    __version__ = 'v1.0.20210316'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)


class GeneralDBDetect(TextDetectOperator):
    """
    避免DB名称与数据库重叠，所以起名DBDetect
    notes:
    v1.0.20210413:增加对于纵横比离谱的图像的适配。避免显存爆炸
    v1.0.20210414:调整score计算方式
    """

    name = '自然场景下的基于DB的文本检测'
    __version__ = 'v1.0.20210414'

    def __init__(self, _inference_helper, _is_test, _threshold=0.3, _bbox_scale_ratio=1.5, _shortest_length=5):
        super().__init__(_inference_helper, _is_test)
        self.threshold = _threshold
        self.bbox_scale_ratio = _bbox_scale_ratio
        self.shortest_length = _shortest_length

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            support_backbone_type = {'mbv3', 'res18'}
            backbone_type = 'res18'
            if 'backbone_type' in self.inference_config and \
                    self.inference_config['backbone_type'] in support_backbone_type:
                backbone_type = self.inference_config['backbone_type']
            else:
                print(f'db use the default backbone:{backbone_type}')
            inference_helper = TritonInferenceHelper(
                f'DB_{backbone_type}',
                self.inference_config['triton_url'],
                self.inference_config['triton_port'],
                f'DB_{backbone_type}',
                1
            )
            inference_helper.add_image_input('INPUT__0', (-1, -1, 3), '识别用的图像',
                                             ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]))
            inference_helper.add_output('OUTPUT__0', (1, -1, -1), 'detect score')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for db text detect not implement")

    def execute(self, _image):
        to_return_result = {
            'locations': [],
        }
        h, w = _image.shape[:2]
        aspect_ratio = max(h, w) / min(h, w)
        bgr_image = force_convert_image_to_bgr(_image)
        need_crop = False
        left_pad, top_pad = 0, 0
        if aspect_ratio < 3:
            resized_image = resize_with_specific_base(resize_with_short_side(bgr_image, max(736, min(h, w))), 32, 32)
            candidate_image = resized_image
        else:
            # 目前测试的最严重的长宽比为30：1
            resized_image = resize_with_long_side(bgr_image, 736)
            candidate_image, (left_pad, top_pad) = center_pad_image_with_specific_base(
                resized_image, 736, 736, _output_pad_ratio=True
            )
            need_crop = True
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            score_map = result['OUTPUT__0']
        else:
            raise NotImplementedError(f"{self.inference_helper.type_name} helper for db not implement")
        if need_crop:
            resized_h, resized_w = resized_image.shape[:2]
            candidate_h, candidate_w = candidate_image.shape[:2]
            start_x = 0
            start_y = 0
            if left_pad != 0:
                start_x = int(left_pad * candidate_w)
            if top_pad != 0:
                start_y = int(top_pad * candidate_h)
            score_map = score_map[..., start_y:start_y + resized_h, start_x:start_x + resized_w]
        boxes, scores = db_post_process(score_map, self.threshold, self.bbox_scale_ratio, self.shortest_length)
        for m_box, m_score in zip(boxes, scores):
            to_return_result['locations'].append({
                'box_info': m_box,
                'score': m_score,
            })
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    import cv2
    from Utils.AnnotationTools import draw_rotated_bbox
    from Utils.GeometryUtils import get_rotated_box_roi_from_image

    ag = ArgumentParser('Text Detect Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    ag.add_argument('-b', '--backbone', dest='backbone', choices=['res18', 'mbv3'], default='res18', help='DB的backbone')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
    db_handler = GeneralDBDetect({
        'name': 'triton',
        'backbone_type': args.backbone,
        'triton_url': args.triton_url,
        'triton_port': args.triton_port},
        True, 0.3, 5, 5
    )
    db_boxes = db_handler.execute(img)['locations']
    db_result_to_show = img.copy()
    for m_box_index, m_box in enumerate(db_boxes, 1):
        draw_rotated_bbox(db_result_to_show, m_box['box_info'], (0, 0, 255), 2)
        m_roi_image = get_rotated_box_roi_from_image(img, _rotated_box=m_box['box_info'])
        cv2.imshow(f'roi No.{m_box_index}', m_roi_image)
    cv2.imshow(f'db_{args.backbone}_result_to_show', db_result_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
