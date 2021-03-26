import os

from Deployment.ConsumerWorker import celery_worker_app
from Deployment.server_config import IS_TEST, FACE_TRITON_URL, FACE_TRITON_PORT
from Operators.ExampleFaceAlignmentOperator import GeneralLandmark106p
from Operators.ExampleFaceLivenessDetect import GeneralMiniFASNetV2
from Operators.ExampleFaceParsingOperator.FaceParsingOperator import GeneralFaceParsing
from Operators.ExampleImageEmbeddingOperator import AsiaFaceEmbedding
from Utils.GeometryUtils import get_rotated_box_roi_from_image
from Utils.ServiceUtils import ServiceTask
from Utils.Storage import get_oss_handler
from Operators.ExampleFaceDetectOperator import GeneralRetinaFaceDetect
from Utils.misc import get_date_string, get_uuid_name

face_detect_op = GeneralRetinaFaceDetect({
    'name': 'triton',
    'triton_url': FACE_TRITON_URL,
    'triton_port': FACE_TRITON_PORT
}, IS_TEST, 0.7, 0.5)
landmark106p_detect_handler = GeneralLandmark106p({
    'name': 'triton',
    'triton_url': FACE_TRITON_URL,
    'triton_port': FACE_TRITON_PORT
}, True)
face_parsing_handler = GeneralFaceParsing({
    'name': 'triton',
    'triton_url': FACE_TRITON_URL,
    'triton_port': FACE_TRITON_PORT
}, True)
asia_face_embedding_handler = AsiaFaceEmbedding({
    'name': 'triton',
    'triton_url': FACE_TRITON_URL,
    'triton_port': FACE_TRITON_PORT
}, True)
mini_fasnetv2_handler = GeneralMiniFASNetV2({
    'name': 'triton',
    'triton_url': FACE_TRITON_URL,
    'triton_port': FACE_TRITON_PORT
}, True)


@celery_worker_app.task(name="ConsumerServices.FaceService.face_detect")
def face_detect(_image_info):
    """
    人脸检测

    Args:
        _image_info:    待识别的完整图像

    Returns:    检测到的人脸区域

    """
    to_return_result = {
        'face_count': 0,
        'face_box_info': []
    }
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    face_detect_result = face_detect_op.execute(img)
    to_return_result['face_count'] = len(face_detect_result['locations'])
    to_return_result['face_box_info'] = face_detect_result['locations']
    return to_return_result


class FaceDetectServiceTask(ServiceTask):
    service_version = 'v1.0.20210326'
    service_name = 'face_detect'
    mock_result = {
        'face_box_info': [],
        'face_count': 0,
    }
    require_field = {
        "_image_info",
    }
    binding_service = face_detect


@celery_worker_app.task(name="ConsumerServices.FaceService.face_landmark")
def face_landmark(_image_info, _face_box_info):
    """
    人脸landmark检测

    Args:
        _image_info:    待识别的完整图像
        _face_box_info:  人脸所在区域

    Returns:    人脸landmark坐标

    """
    to_return_result = {
        'points_count': 106,
        'x_locations': [0] * 106,
        'y_locations': [0] * 106,
    }
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    cropped_image = get_rotated_box_roi_from_image(img, _face_box_info, 1.35)
    landmark_detect_result = landmark106p_detect_handler.execute(cropped_image)
    to_return_result = landmark_detect_result.copy()
    return to_return_result


class FaceLandmarkServiceTask(ServiceTask):
    service_version = 'v1.0.20210326'
    service_name = 'face_landmark'
    mock_result = {
        'points_count': 106,
        'x_locations': [0] * 106,
        'y_locations': [0] * 106,
    }
    require_field = {
        "_image_info",
        "_face_box_info"
    }
    binding_service = face_landmark


@celery_worker_app.task(name="ConsumerServices.FaceService.face_parsing")
def face_parsing(_image_info, _face_box_info, _face_landmark_info):
    """
    人脸语义分区

    Args:
        _image_info:    待识别的完整图像
        _face_box_info:  人脸所在区域
        _face_landmark_info:    人脸landmark坐标信息

    Returns:    人脸不同区域的mask的key

    """
    to_return_result = {
        'parsing_info': {
            'bucket_name': '',
            'path': ''
        },
    }
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    cropped_image = get_rotated_box_roi_from_image(img, _face_box_info, _scale_ratio=1.5)
    face_parsing_result = face_parsing_handler.execute(cropped_image, _face_landmark_info)
    parsing_result = face_parsing_result['semantic_segmentation']
    date_string = get_date_string()
    name_string = get_uuid_name()
    target_path = os.path.join(date_string, name_string)
    target_path = oss_handler.upload_numpy_array('intermediate', target_path, parsing_result)
    to_return_result['parsing_info']['bucket_name'] = 'intermediate'
    to_return_result['parsing_info']['path'] = target_path
    return to_return_result


class FaceParsingServiceTask(ServiceTask):
    service_version = 'v1.0.20210326'
    service_name = 'face_parsing'
    mock_result = {
        'parsing_info': {
            'bucket_name': '',
            'path': ''
        },
    }
    require_field = {
        "_image_info",
        "_face_box_info",
        "_face_landmark_info",
    }
    binding_service = face_parsing


@celery_worker_app.task(name="ConsumerServices.FaceService.face_embedding")
def face_embedding(_image_info, _face_box_info, _face_landmark_info):
    """
    人脸特征向量提取

    Args:
        _image_info:    待识别的完整图像
        _face_box_info:  人脸所在区域
        _face_landmark_info:    人脸landmark坐标信息

    Returns:    人脸的特征向量

    """
    to_return_result = {"face_feature_vector": []}
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    cropped_image = get_rotated_box_roi_from_image(img, _face_box_info, 1.35)
    embedding_result = asia_face_embedding_handler.execute(cropped_image, _face_landmark_info)
    to_return_result['face_feature_vector'] = embedding_result['feature_vector']
    return to_return_result


class FaceEmbeddingServiceTask(ServiceTask):
    service_version = 'v1.0.20210326'
    service_name = 'face_embedding'
    mock_result = {
        'face_feature_vector': []
    }
    require_field = {
        "_image_info",
        "_face_box_info",
        "_face_landmark_info",
    }
    binding_service = face_embedding


@celery_worker_app.task(name="ConsumerServices.FaceService.face_liveness_detect")
def face_liveness_detect(_image_info, _face_box_info):
    """
    静默人脸活体检测

    Args:
        _image_info:    待识别的完整图像
        _face_box_info:  人脸所在区域

    Returns:    人脸的特征向量

    """
    to_return_result = {"is_fake": False}
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    cropped_image = get_rotated_box_roi_from_image(img, _face_box_info, 2.7)
    liveness_result = mini_fasnetv2_handler.execute(cropped_image)
    score = liveness_result['classification_scores']
    to_return_result['is_fake'] = (score[1] > score[0]) and (score[1] > score[2])
    return to_return_result


class FaceLivenessDetectServiceTask(ServiceTask):
    service_version = 'v1.0.20210326'
    service_name = 'face_liveness_detect'
    mock_result = {
        'is_fake': False
    }
    require_field = {
        "_image_info",
        "_face_box_info",
    }
    binding_service = face_liveness_detect
