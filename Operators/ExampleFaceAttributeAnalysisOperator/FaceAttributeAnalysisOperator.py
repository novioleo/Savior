from enum import Enum

from scipy.special import softmax

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Utils.GeometryUtils import face_align, center_pad_image_with_specific_base, force_convert_image_to_bgr
from Utils.InferenceHelpers import TritonInferenceHelper
import numpy as np
import cv2


class RaceType(Enum):
    WHITE: int = 0
    BLACK: int = 1
    LATINO_HISPANIC: int = 2
    EAST_ASIAN: int = 3
    SOUTHEAST_ASIAN: int = 4
    INDIAN: int = 5
    MIDDLE_EASTERN: int = 6


class SexualType(Enum):
    MALE: int = 0
    FEMALE: int = 1


class AgeRaceGenderWithFair(DummyAlgorithmWithModel):
    name = 'Fair'
    __version__ = 'v1.0.20210409'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper('Fair',
                                                     self.inference_config['triton_url'],
                                                     self.inference_config['triton_port'],
                                                     'Fair', 1)
            inference_helper.add_image_input('INPUT__0', (224, 224, 3), '人脸图像',
                                             ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]))
            inference_helper.add_output('OUTPUT__0', (18,), 'race[:7] sexual[7:9] age[9:18]')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for fair not implement")

    def execute(self, _image, _landmark_info=None):
        to_return_result = {
            'age_lower_boundary': 0,
            'age_higher_boundary': 10,
            'race_type': RaceType.EAST_ASIAN,
            'sexual': SexualType.MALE,
        }
        lower_boundaries = [0, 3, 10, 20, 30, 40, 50, 60, 70]
        higher_boundaries = [2, 9, 19, 29, 39, 49, 59, 69, 100]
        race_list = [
            RaceType.WHITE,
            RaceType.BLACK,
            RaceType.LATINO_HISPANIC,
            RaceType.EAST_ASIAN,
            RaceType.SOUTHEAST_ASIAN,
            RaceType.INDIAN,
            RaceType.MIDDLE_EASTERN
        ]
        sexual_list = [
            SexualType.MALE,
            SexualType.FEMALE
        ]
        if _landmark_info is not None:
            if _landmark_info['points_count'] == 0:
                candidate_index = list(range(5))
            elif _landmark_info['points_count'] == 106:
                candidate_index = [38, 88, 86, 52, 61]
            else:
                raise NotImplementedError(
                    f"Cannot align face with {_landmark_info['points_count']} landmark points now")
            landmark_x = _landmark_info['x_locations'][candidate_index]
            landmark_y = _landmark_info['y_locations'][candidate_index]
            landmark = np.stack([landmark_x, landmark_y], axis=1)
            aligned_face = face_align(_image, landmark, (192, 224))
        else:
            aligned_face = cv2.resize(_image, (192, 224))
        padded_face = center_pad_image_with_specific_base(aligned_face, 224, 224, 0, False)
        candidate_image = force_convert_image_to_bgr(padded_face)
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            classification_result = result['OUTPUT__0'].squeeze(0)
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for fair not implement")
        race_index = np.argmax(softmax(classification_result[:7], axis=0))
        gender_index = np.argmax(softmax(classification_result[7:9], axis=0))
        age_index = np.argmax(softmax(classification_result[9:18], axis=0))
        to_return_result = {
            'age_lower_boundary': lower_boundaries[age_index],
            'age_higher_boundary': higher_boundaries[age_index],
            'race_type': race_list[race_index],
            'sexual': sexual_list[gender_index],
        }
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    from Utils.GeometryUtils import get_rotated_box_roi_from_image
    from Operators.ExampleFaceDetectOperator import GeneralUltraLightFaceDetect
    from Operators.ExampleFaceAlignmentOperator import GeneralLandmark106p
    from pprint import pprint

    ag = ArgumentParser('Face Attribute Analysis Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='人物照片路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
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
    fair_handler = AgeRaceGenderWithFair({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port,
    }, True)
    # 假设图中有且仅有一人
    all_features = []
    face_detect_result = ultra_light_face_detect_handler.execute(img)
    face_bbox_info = face_detect_result['locations'][0]['box_info']
    cropped_face = get_rotated_box_roi_from_image(img, face_bbox_info, 1.25)
    landmark_result = landmark106p_detect_handler.execute(cropped_face)
    attribute_result = fair_handler.execute(cropped_face, landmark_result)
    pprint(attribute_result)
