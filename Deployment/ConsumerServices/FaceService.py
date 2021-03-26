from Deployment.ConsumerWorker import celery_worker_app
from Utils.GeometryUtils import get_rotated_box_roi_from_image
from Utils.ServiceUtils import ServiceTask
from Utils.Storage import get_oss_handler


@celery_worker_app.task(name="ConsumerServices.FaceService.face_detect")
def face_detect(_image_info):
    """
    人脸检测

    Args:
        _image_info:    待识别的完整图像

    Returns:    检测到的人脸区域

    """
    to_return_result = {'face_box_info': []}
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    return to_return_result


class FaceDetectServiceTask(ServiceTask):
    service_version = 'v1.0.20210318'
    service_name = 'face_detect'
    mock_result = {
        'face_box_info': [],
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
    to_return_result = {'landmark_info': []}
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    cropped_image = get_rotated_box_roi_from_image(img, _face_box_info)
    return to_return_result


class FaceLandmarkServiceTask(ServiceTask):
    service_version = 'v1.0.20210318'
    service_name = 'face_landmark'
    mock_result = {
        'landmark_info': [],
    }
    require_field = {
        "_image_info",
        "_face_box_info"
    }
    binding_service = face_detect


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
            'left_eye': '',
            'right_eye': '',
            'nose': '',
            'mouth': '',
            'left_eyebrow': '',
            'right_eyebrow': ''
        },
    }
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    cropped_image = get_rotated_box_roi_from_image(img, _face_box_info)
    return to_return_result


class FaceParsingServiceTask(ServiceTask):
    service_version = 'v1.0.20210318'
    service_name = 'face_parsing'
    mock_result = {
        'parsing_info': {
            'left_eye': '',
            'right_eye': '',
            'nose': '',
            'mouth': '',
            'left_eyebrow': '',
            'right_eyebrow': ''
        },
    }
    require_field = {
        "_image_info",
        "_face_box_info",
        "_face_landmark_info",
    }
    binding_service = face_detect


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
    to_return_result = {"face_feature": []}
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    cropped_image = get_rotated_box_roi_from_image(img, _face_box_info)
    return to_return_result


class FaceEmbeddingServiceTask(ServiceTask):
    service_version = 'v1.0.20210318'
    service_name = 'face_embedding'
    mock_result = {
        'face_feature': []
    }
    require_field = {
        "_image_info",
        "_face_box_info",
        "_face_landmark_info",
    }
    binding_service = face_detect
