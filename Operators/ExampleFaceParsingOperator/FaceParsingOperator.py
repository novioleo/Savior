from abc import ABC

import cv2
import numpy as np

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Utils.GeometryUtils import center_pad_image_with_specific_base, \
    resize_with_long_side, force_convert_image_to_bgr, correct_face_orientation
from Utils.InferenceHelpers import TritonInferenceHelper


class FaceParsingOperator(DummyAlgorithmWithModel, ABC):
    name = 'FaceParsing'
    __version__ = 'v1.0.20210323'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)


class GeneralFaceParsing(FaceParsingOperator):
    """
    获取人脸面部分区，除了面部区域，其他地方准确率很低
    例如耳环、眼镜等
    """
    name = '自然场景下基于BiSeNet人脸面部的语义分割'
    __version__ = 'v1.0.20210323'

    def __init__(self, _inference_config, _is_test):
        """
        每个下标对应的意思
        0   背景
        1   皮肤区域
        2   右眉毛
        3   左眉毛
        4   右眼睛
        5   左眼睛
        6   眼镜
        7   右耳朵
        8   左耳朵
        9   耳环
        10  鼻子
        11  口腔
        12  上嘴唇
        13  下嘴唇
        14  颈部
        15
        16  衣服
        17  头发
        18  帽子
        """
        super().__init__(_inference_config, _is_test)
        # 模型未限制，但是为了保证效率，将图像都统一到512
        self.candidate_image_size = (512, 512)

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper('FaceParsing',
                                                     self.inference_config['triton_url'],
                                                     self.inference_config['triton_port'],
                                                     'FaceParsing', 1)
            inference_helper.add_image_input('INPUT__0', (512, 512, 3), '检测用rgb的图像',
                                             ([103.53, 116.28, 123.675], [57.375, 57.12, 58.395]))
            inference_helper.add_output('OUTPUT__0', (19, 512, 512), '每个类别的区域')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for face parsing not implement")

    def execute(self, _image, _landmark_info=None):
        to_return_result = {
            'semantic_segmentation': np.zeros((_image.shape[1], _image.shape[0]), dtype=np.uint8),
        }
        if _landmark_info is not None:
            corrected_face_image, rotate_back_function = correct_face_orientation(_image, _landmark_info)
        else:
            corrected_face_image = _image

            def _rotate_back_function(_image):
                return _image

            rotate_back_function = _rotate_back_function
        original_h, original_w = corrected_face_image.shape[:2]
        resized_image = resize_with_long_side(corrected_face_image, 512)
        resized_h, resized_w = resized_image.shape[:2]
        padded_image, (width_pad_ratio, height_pad_ratio) = center_pad_image_with_specific_base(
            resized_image,
            _width_base=512,
            _height_base=512,
            _output_pad_ratio=True
        )
        candidate_image = cv2.cvtColor(force_convert_image_to_bgr(padded_image), cv2.COLOR_BGR2RGB)
        candidate_h, candidate_w = candidate_image.shape[:2]
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            semantic_index = result['OUTPUT__0'].squeeze()
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for face parsing not implement")
        left_width_pad = int(width_pad_ratio * candidate_w)
        top_height_pad = int(height_pad_ratio * candidate_h)
        # 去除pad
        semantic_index_without_pad = semantic_index[
                                     top_height_pad:top_height_pad + resized_h,
                                     left_width_pad:left_width_pad + resized_w
                                     ]
        # 恢复resize
        resize_back_semantic_index = cv2.resize(semantic_index_without_pad, (original_w, original_h),
                                                interpolation=cv2.INTER_NEAREST)
        # 恢复图像方向
        original_orientation_semantic_index = rotate_back_function(resize_back_semantic_index)
        to_return_result['semantic_segmentation'] = original_orientation_semantic_index
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    from Utils.AnnotationTools import annotate_segmentation
    from Operators.ExampleFaceDetectOperator import GeneralUltraLightFaceDetect
    from Operators.ExampleFaceAlignmentOperator import GeneralLandmark106p
    from Utils.GeometryUtils import get_rotated_box_roi_from_image

    ag = ArgumentParser('Face Parsing Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    # 假设图中只有一个人头
    img = cv2.imread(args.image_path)
    face_parsing_handler = GeneralFaceParsing({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True)
    ultra_light_face_detect_handler = GeneralUltraLightFaceDetect({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True, 0.7, 0.5)
    landmark106p_detect_handler = GeneralLandmark106p({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True)
    face_bbox = ultra_light_face_detect_handler.execute(img)['locations'][0]['box_info']
    cropped_image = get_rotated_box_roi_from_image(img, face_bbox, 1.35)
    landmark_info = landmark106p_detect_handler.execute(cropped_image)
    landmark106p_with_bbox_result_image = cropped_image.copy()
    landmark106p_with_bbox_result_all_points = [(x, y) for x, y in
                                                zip(landmark_info['x_locations'],
                                                    landmark_info['y_locations'])
                                                ]
    face_parsing_with_bbox_result = face_parsing_handler.execute(cropped_image, landmark_info)
    face_parsing_with_bbox_result_image = cropped_image.copy()
    face_parsing_with_bbox_result_image = annotate_segmentation(
        face_parsing_with_bbox_result_image,
        face_parsing_with_bbox_result['semantic_segmentation']
    )
    cv2.imshow(f'face_parsing_with_bbox_result_image', face_parsing_with_bbox_result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
