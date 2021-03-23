from abc import ABC
import numpy as np
import cv2
from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Utils.InferenceHelpers import TritonInferenceHelper


class FaceEmbeddingOperator(DummyAlgorithmWithModel, ABC):
    name = 'FaceEmbedding'
    __version__ = 'v1.0.20210322'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)
        self.reference_facial_points = np.array([
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 92.3655014],
            [62.72990036, 92.20410156]
        ], dtype=np.float32)

    def face_align(self, _cropped_image, _landmark):
        h, w = _cropped_image.shape[:2]
        remapped_landmark = _landmark.copy()
        remapped_landmark[:, 0] *= w
        remapped_landmark[:, 1] *= h
        transform_matrix = cv2.estimateRigidTransform(remapped_landmark, self.reference_facial_points, True)
        face_img = cv2.warpAffine(_cropped_image, transform_matrix, (96, 112))
        return face_img


class AsiaFaceEmbedding(FaceEmbeddingOperator):
    name = '基于IRes50的亚洲人脸的embedding'
    __version__ = 'v1.0.20210322'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)
        self.candidate_size = (112, 112)

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper(
                'AsiaFaceEmbedding',
                self.inference_config['triton_url'],
                self.inference_config['triton_port'],
                'AsiaFaceEmbedding',
                1
            )
            inference_helper.add_image_input('INPUT__0', (112, 112, 3), '人脸图像',
                                             ([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]))
            inference_helper.add_output('OUTPUT__0', (1, 512), '人脸特征向量')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for asian face embedding not implement")

    def execute(self, _image, _landmark_info):
        to_return_result = {
            'feature_vector': [0, ] * 512,
        }
        if _landmark_info['points_count'] == 0:
            candidate_index = list(range(5))
        elif _landmark_info['points_count'] == 106:
            candidate_index = [38, 88, 86, 52, 61]
        else:
            raise NotImplementedError(f"Cannot align face with {_landmark_info['points_count']} landmark points now")
        landmark_x = _landmark_info['x_locations'][candidate_index]
        landmark_y = _landmark_info['y_locations'][candidate_index]
        landmark = np.stack([landmark_x, landmark_y], axis=1)
        aligned_face = self.face_align(_image, landmark)
        padded_face = center_pad_image_with_specific_base(aligned_face, 112, 112, 0, False)
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=padded_face.astype(np.float32))
            face_feature_vector = result['OUTPUT__0'].squeeze()
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for asian face embedding not implement")
        to_return_result['feature_vector'] = face_feature_vector.tolist()
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    from Utils.GeometryUtils import get_rotated_box_roi_from_image, center_pad_image_with_specific_base
    from Operators.ExampleFaceDetectOperator import GeneralUltraLightFaceDetect
    from Operators.ExampleFaceAlignmentOperator import GeneralLandmark106p
    from sklearn.metrics import pairwise_distances

    ag = ArgumentParser('Face Embedding Example')
    ag.add_argument('--anchor_image_path', dest='anchor_image_path', type=str, required=True, help='人物A照片路径')
    ag.add_argument('--positive_image_path', dest='positive_image_path', type=str, required=True, help='人物A另一张照片路径')
    ag.add_argument('--negative_image_path', dest='negative_image_path', type=str, required=True, help='人物B照片路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    anchor_img = cv2.imread(args.anchor_image_path)
    positive_img = cv2.imread(args.positive_image_path)
    negative_img = cv2.imread(args.negative_image_path)
    asia_face_embedding_handler = AsiaFaceEmbedding({
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
    # 假设图中有且仅有一人
    all_features = []
    for m_img in [anchor_img, positive_img, negative_img]:
        m_face_detect_result = ultra_light_face_detect_handler.execute(m_img)
        m_face_bbox_info = m_face_detect_result['locations'][0]['box_info']
        m_cropped_face = get_rotated_box_roi_from_image(m_img, m_face_bbox_info, 1.25)
        m_landmark_result = landmark106p_detect_handler.execute(m_cropped_face)
        m_embedding_result = asia_face_embedding_handler.execute(m_cropped_face, m_landmark_result)
        all_features.append(m_embedding_result['feature_vector'])
    all_features_np = np.array(all_features, dtype=np.float32)
    distance_matrix = pairwise_distances(all_features_np, metric='euclidean')
    print('anchor-positive distance:',distance_matrix[0][1])
    print('anchor-negative distance:',distance_matrix[0][2])
    print('positive-negative distance:',distance_matrix[1][2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
