from abc import ABC

from Operators.DummyAlgorithm import DummyAlgorithm
import cv2
import numpy as np

from Operators.ExampleTextDetectOperator.PostProcess import db_post_process
from Utils.GeometryUtils import resize_with_short_side, resize_with_specific_base
from Utils.InferenceHelpers.BaseInferenceHelper import TritonInferenceHelper


class TextDetectOperator(DummyAlgorithm, ABC):
    name = 'TextDetect'
    __version__ = 'v1.0.20210316'

    def __init__(self, _inference_helper, _is_test):
        super().__init__(_is_test)
        self.inference_helper = _inference_helper


class GeneralDBDetect(TextDetectOperator):
    """
    避免DB名称与数据库重叠，所以起名DBDetect
    """
    name = '自然场景下的基于DB的文本检测'
    __version__ = 'v1.0.20210316'

    def __init__(self, _inference_helper, _is_test, _threshold=0.3, _bbox_scale_ratio=1.5, _shortest_length=5):
        super().__init__(_inference_helper, _is_test)
        self.threshold = _threshold
        self.bbox_scale_ratio = _bbox_scale_ratio
        self.shortest_length = _shortest_length

    def execute(self, _image):
        to_return_result = {
            'locations': [],
        }
        h, w = _image.shape[:2]
        resized_image = resize_with_specific_base(resize_with_short_side(_image, max(736, min(h, w))), 32, 32)
        if len(resized_image.shape) == 2:
            candidate_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        else:
            if resized_image.shape[-1] == 4:
                candidate_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2BGR)
            else:
                candidate_image = resized_image
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            score_map = result['OUTPUT__0']
        else:
            raise NotImplementedError(f"{self.inference_helper.type_name} helper for db not implement")
        boxes, scores = db_post_process(score_map, self.threshold, self.bbox_scale_ratio, self.shortest_length)
        for m_box, m_score in zip(boxes, scores):
            to_return_result['locations'].append({
                'box_info': m_box,
                'score': m_score,
            })
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    from Utils.AnnotationTools import draw_rotated_bbox

    ag = ArgumentParser('Text Detect Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
    db_mbv3 = TritonInferenceHelper('DB_mbv3', args.triton_url, args.triton_port, 'DB_mbv3', 1)
    db_mbv3.add_image_input('INPUT__0', (-1, -1, 3), '识别用的图像', ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]))
    db_mbv3.add_output('OUTPUT__0', (1, -1, -1), 'detect score')
    db_res18 = TritonInferenceHelper('DB_res18', args.triton_url, args.triton_port, 'DB_res18', 1)
    db_res18.add_image_input('INPUT__0', (-1, -1, 3), '识别用的图像', ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]))
    db_res18.add_output('OUTPUT__0', (1, -1, -1), 'detect score')
    db_mbv3_handler = GeneralDBDetect(db_mbv3, True, 0.3, 1.5, 5)
    db_res18_handler = GeneralDBDetect(db_res18, True, 0.3, 5, 5)
    db_mbv3_boxes = db_mbv3_handler.execute(img)['locations']
    db_res18_boxes = db_res18_handler.execute(img)['locations']
    db_mbv3_result_to_show = img.copy()
    for m_box in db_mbv3_boxes:
        draw_rotated_bbox(db_mbv3_result_to_show, m_box['box_info'], (255, 0, 0), 2)
    cv2.imshow('db_mbv3_result_to_show', db_mbv3_result_to_show)
    db_res18_result_to_show = img.copy()
    for m_box in db_res18_boxes:
        draw_rotated_bbox(db_res18_result_to_show, m_box['box_info'], (0, 0, 255), 2)
    cv2.imshow('db_res18_result_to_show', db_res18_result_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
