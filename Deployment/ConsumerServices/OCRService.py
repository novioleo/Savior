import os

from Deployment.ConsumerWorker import celery_worker_app
from Deployment.server_config import IS_TEST, TRITON_URL, TRITON_PORT
from Operators.ExampleTextDetectOperator.TextDetectOperator import GeneralDBDetect
from Operators.ExampleTextRecognizeOperator.TextRecognizeOperator import GeneralCRNN
from Utils.AnnotationTools import annotate_detect_rotated_bbox_and_text_result
from Utils.GeometryUtils import get_rotated_box_roi_from_image
from Utils.InferenceHelpers.BaseInferenceHelper import TritonInferenceHelper
from Utils.ServiceUtils import ServiceTask
from Utils.Storage import get_oss_handler
from Utils.misc import get_date_string, get_uuid_name

# 初始化所有会用到的op
# 初始化crnn的op
crnn_res34_triton_helper = TritonInferenceHelper('crnn_res34', TRITON_URL, TRITON_PORT, 'CRNN_res34', 1)
crnn_res34_triton_helper.add_image_input('INPUT__0', (32, -1, 3), '识别用的图像',
                                         ([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]))
crnn_res34_triton_helper.add_output('OUTPUT__0', (-1, 1), '识别的max')
crnn_res34_triton_helper.add_output('OUTPUT__1', (-1, 1), '识别的argmax的结果')
text_recognize_op = GeneralCRNN(crnn_res34_triton_helper, 'common', IS_TEST)
# 初始化db的op
db_res18 = TritonInferenceHelper('DB_res18', TRITON_URL, TRITON_PORT, 'DB_res18', 1)
db_res18.add_image_input('INPUT__0', (-1, -1, 3), '识别用的图像', ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]))
db_res18.add_output('OUTPUT__0', (1, -1, -1), 'detect score')
db_res18_op = GeneralDBDetect(db_res18, True, 0.3, 5, 5)


@celery_worker_app.task(name="ConsumerServices.OCRService.text_recognize")
def text_recognize(_image_info, _box_info):
    """
    文本识别

    Args:
        _image_info:    待识别的完整图像
        _box_info:      图像中文本区域的位置

    Returns:    文本区域位置的识别结果

    """
    to_return_result = {'text': ''}
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    cropped_image = get_rotated_box_roi_from_image(img, _box_info)
    recognize_result = text_recognize_op.execute(cropped_image)
    to_return_result['text'] = recognize_result['text']
    return to_return_result


class TextRecognizeServiceTask(ServiceTask):
    service_version = 'v1.0.20210315'
    service_name = 'text_recognize'
    mock_result = {
        'text': '',
    }
    require_field = {
        "_image_info",
        "_box_info",
    }
    binding_service = text_recognize


@celery_worker_app.task(name="ConsumerServices.OCRService.text_detect")
def text_detect(_image_info):
    """
    文本检测

    Args:
        _image_info:    待检测的图像信息

    Returns:    检测得到的所有box

    """
    to_return_result = {'box_info': [], 'box_count': 0}
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    detect_result = db_res18_op.execute(img)
    for m_box in detect_result['locations']:
        m_box_info = m_box['box_info']
        m_box_score = m_box['score']
        to_return_result['box_info'].append({
            'degree': m_box_info['degree'],
            'center_x': m_box_info['center_x'],
            'center_y': m_box_info['center_y'],
            'box_height': m_box_info['box_height'],
            'box_width': m_box_info['box_width'],
            'score': m_box_score,
        })
    to_return_result['box_count'] = len(detect_result['locations'])
    return to_return_result


class TextDetectServiceTask(ServiceTask):
    service_version = 'v1.0.20210315'
    service_name = 'text_detect'
    mock_result = {
        'box_info': [
            {
                'degree': 0,
                'center_x': 0.23,
                'center_y': 0.17,
                'box_height': 0.22,
                'box_width': 0.55,
                'score': 0.98,
            },
        ],
        'box_count': 1
    }
    require_field = {
        "_image_info",
    }
    binding_service = text_detect


@celery_worker_app.task(name="ConsumerServices.OCRService.ocr_result_visualization")
def ocr_result_visualization(_image_info, _box_info_list, _text_list):
    """
    将检测的结果和识别的结果合并到一张图中

    Args:
        _image_info:    图片信息
        _box_info_list:     所有box的列表
        _text_list:     跟box顺序一致的识别结果

    Returns:    合并后的图片的oss的路径

    """
    to_return_result = {'bucket_name': '', 'path': ''}
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    result_image = annotate_detect_rotated_bbox_and_text_result(img, _box_info_list, _text_list, (0, 0, 255), 3)
    date_string = get_date_string()
    uuid_name = get_uuid_name()
    image_path = os.path.join(date_string, uuid_name)
    final_image_path = oss_handler.upload_image_file('result', image_path, result_image, True, 50)
    to_return_result['bucket_name'] = 'result'
    to_return_result['path'] = final_image_path
    return to_return_result


class OCRResultVisualizationServiceTask(ServiceTask):
    service_version = 'v1.0.20210317'
    service_name = 'ocr_result_visualization'
    mock_result = {'bucket_name': 'result', 'path': 'fake/path.webp'}
    require_field = {
        "_image_info",
        "_box_info_list",
        "_text_list"
    }
    binding_service = ocr_result_visualization
