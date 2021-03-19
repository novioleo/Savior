from abc import ABC

import cv2
import numpy as np

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Operators.ExampleFaceDetectOperator.PostProcessUtils import get_anchors, regress_boxes
from Utils.GeometryUtils import center_pad_image_with_specific_base, \
    nms, resize_with_long_side, force_convert_image_to_bgr
from Utils.InferenceHelpers.BaseInferenceHelper import TritonInferenceHelper


class FaceDetectOperator(DummyAlgorithmWithModel, ABC):
    name = 'FaceDetect'
    __version__ = 'v1.0.20210319'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)


class GeneralUltraLightFaceDetect(FaceDetectOperator):
    """
    这个模型超级轻量级，速度快，但是漏检的概率会比较大
    适合对检测精度要求不高的场景，人脸比较明显的场景
    """
    name = '自然场景下的基于UltraLightFaceDetect的人脸检测'
    __version__ = 'v1.0.20210319'

    def __init__(self, _inference_config, _is_test, _score_threshold=0.7, _iou_threshold=0.5):
        super().__init__(_inference_config, _is_test)
        self.score_threshold = _score_threshold
        self.iou_threshold = _iou_threshold
        self.candidate_image_size = (320, 240)

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper('UltraLightFaceDetect',
                                                     self.inference_config['triton_url'],
                                                     self.inference_config['triton_port'],
                                                     'UltraLightFaceDetect', 1)
            inference_helper.add_image_input('INPUT__0', (320, 240, 3), '识别用的图像',
                                             ([127, 127, 127], [128, 128, 128]))
            inference_helper.add_output('OUTPUT__0', (1, 4420, 2), 'detect score')
            inference_helper.add_output('OUTPUT__1', (1, 4420, 4), 'box predict')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for ultra light face detect not implement")

    def execute(self, _image):
        to_return_result = {
            'locations': [],
        }
        padded_image, (width_pad_ratio, height_pad_ratio) = center_pad_image_with_specific_base(
            _image,
            _width_base=32,
            _height_base=24,
            _output_pad_ratio=True
        )
        resized_image = cv2.resize(_image, self.candidate_image_size)
        candidate_image = force_convert_image_to_bgr(resized_image)
        if isinstance(self.inference_helper, TritonInferenceHelper):
            rgb_image = cv2.cvtColor(candidate_image, cv2.COLOR_BGR2RGB)
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=rgb_image.astype(np.float32))
            score_map = result['OUTPUT__0'].squeeze()
            box = result['OUTPUT__1'].squeeze()
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for ultra light face detect not implement")
        # 0为bg，1为人脸
        box_score_map = score_map[..., 1]
        available_box = box_score_map > self.score_threshold
        if np.sum(available_box) == 0:
            return to_return_result
        filter_scores = box_score_map[available_box]
        filtered_box = box[available_box, :]
        final_box_index = nms(filtered_box, filter_scores, _nms_threshold=self.iou_threshold)
        final_boxes = filtered_box[final_box_index]
        final_scores = filter_scores[final_box_index]
        for m_box, m_score in zip(final_boxes, final_scores):
            m_box_width = m_box[2] - m_box[0]
            m_box_height = m_box[3] - m_box[1]
            m_box_center_x = m_box[0] + m_box_width / 2 - width_pad_ratio
            m_box_center_y = m_box[1] + m_box_height / 2 - height_pad_ratio
            box_info = {
                'degree': 0,
                'center_x': m_box_center_x,
                'center_y': m_box_center_y,
                'box_height': m_box_height,
                'box_width': m_box_width,
            }
            to_return_result['locations'].append({
                'box_info': box_info,
                'score': m_score,
            })
        return to_return_result


class GeneralRetinaFaceDetect(FaceDetectOperator):
    """
    这个模型比较大，精度比较高
    """
    name = '自然场景下的基于RetinaFace的人脸检测'
    __version__ = 'v1.0.20210319'

    def __init__(self, _inference_config, _is_test, _score_threshold=0.7, _iou_threshold=0.5):
        super().__init__(_inference_config, _is_test)
        self.score_threshold = _score_threshold
        self.iou_threshold = _iou_threshold
        self.candidate_image_size = (512, 512)

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper('RetinaFace',
                                                     self.inference_config['triton_url'],
                                                     self.inference_config['triton_port'],
                                                     'RetinaFace', 1)
            inference_helper.add_image_input('INPUT__0', (512, 512, 3), '识别用的图像',
                                             ([0, 0, 0], [1, 1, 1]))
            inference_helper.add_output('OUTPUT__0', (1, 16128, 2), 'face classification')
            inference_helper.add_output('OUTPUT__1', (1, 16128, 4), 'box predict')
            inference_helper.add_output('OUTPUT__2', (1, 16128, 10), 'landmark')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for retina face detect not implement")

    def execute(self, _image):
        to_return_result = {
            'locations': [],
        }
        resized_image = resize_with_long_side(_image, 512)
        resized_shape = resized_image.shape[:2]
        padded_image, (width_pad_ratio, height_pad_ratio) = center_pad_image_with_specific_base(
            resized_image,
            _width_base=512,
            _height_base=512,
            _output_pad_ratio=True
        )
        candidate_image = force_convert_image_to_bgr(padded_image)
        candidate_shape = candidate_image.shape[:2]
        if isinstance(self.inference_helper, TritonInferenceHelper):
            rgb_image = cv2.cvtColor(candidate_image, cv2.COLOR_BGR2RGB)
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=rgb_image.astype(np.float32))
            filter_scores = result['OUTPUT__0'].squeeze()
            box = result['OUTPUT__1'].squeeze()
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for retina face detect not implement")
        anchors = get_anchors(np.array(candidate_image.shape[:2]))
        all_boxes, _ = regress_boxes(anchors, box, None, candidate_image.shape[:2])
        exp_box_score = np.exp(filter_scores)
        face_classification_index = np.argmax(exp_box_score, axis=-1)
        max_classification_score = np.max(exp_box_score, axis=-1)
        candidate_box_index = (face_classification_index == 0) & (max_classification_score > self.score_threshold)
        filter_scores = max_classification_score[candidate_box_index]
        filtered_box = all_boxes[candidate_box_index]
        if len(filter_scores) == 0:
            return to_return_result
        final_box_index = nms(filtered_box, filter_scores, _nms_threshold=self.iou_threshold)
        final_boxes = filtered_box[final_box_index]
        final_scores = filter_scores[final_box_index]
        for m_box, m_score in zip(final_boxes, final_scores):
            m_box_width = m_box[2] - m_box[0]
            m_box_height = m_box[3] - m_box[1]
            m_box_center_x = (m_box[0] + m_box_width / 2 - width_pad_ratio) * candidate_shape[1] / resized_shape[1]
            m_box_center_y = (m_box[1] + m_box_height / 2 - height_pad_ratio) * candidate_shape[0] / resized_shape[0]
            box_info = {
                'degree': 0,
                'center_x': m_box_center_x,
                'center_y': m_box_center_y,
                'box_height': m_box_height * candidate_shape[0] / resized_shape[0],
                'box_width': m_box_width * candidate_shape[1] / resized_shape[1],
            }
            to_return_result['locations'].append({
                'box_info': box_info,
                'score': m_score,
            })
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    from Utils.AnnotationTools import draw_rotated_bbox

    ag = ArgumentParser('Face Detect Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    ultra_light_face_detect_handler = GeneralUltraLightFaceDetect({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True, 0.7, 0.5)
    retina_face_detect_handler = GeneralRetinaFaceDetect({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True, 0.7, 0.5)
    ultra_light_face_detected_boxes = ultra_light_face_detect_handler.execute(img)['locations']

    ultra_light_face_result = img.copy()
    for m_box in ultra_light_face_detected_boxes:
        draw_rotated_bbox(ultra_light_face_result, m_box['box_info'], (255, 0, 0), 2)
    cv2.imshow('ultra_light_face_result', ultra_light_face_result)

    retina_detected_face_boxes = retina_face_detect_handler.execute(img)['locations']
    retina_detected_face_result = img.copy()
    for m_box in retina_detected_face_boxes:
        draw_rotated_bbox(retina_detected_face_result, m_box['box_info'], (0, 0, 255), 2)
    cv2.imshow('retina_detected_face_result', retina_detected_face_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
