from Deployment.ConsumerWorker import celery_worker_app
from Operators.ExampleImageStringParseOperator import ImageParseFromBase64
from Operators.ExampleTextRecognizeOperator import Captcha1RecognizeWithMaster
from Deployment.server_config import RECAPTCHA_TRITON_URL, RECAPTCHA_TRITON_PORT, IS_TEST
from Utils.ServiceUtils import ServiceTask
from Utils.Storage import get_oss_handler

captcha1_recognize_handler = Captcha1RecognizeWithMaster({
    'name': 'triton',
    'triton_url': RECAPTCHA_TRITON_URL,
    'triton_port': RECAPTCHA_TRITON_PORT}, 'keyboard', IS_TEST
)
base64_image_decode_op = ImageParseFromBase64(IS_TEST)


@celery_worker_app.task(name="ConsumerServices.RecaptchaService.captcha1_recognize")
def captcha1_recognize(_image_info):
    """
    验证码种类1的识别

    Args:
        _image_info:    待识别的完整图像

    Returns:    文本区域位置的识别结果

    """
    to_return_result = {'text': ''}
    oss_handler = get_oss_handler()
    img = oss_handler.download_image_file(
        _image_info['bucket_name'],
        _image_info['path']
    )
    recognize_result = captcha1_recognize_handler.execute(img)
    to_return_result['text'] = recognize_result['text']
    return to_return_result


class Captcha1RecognizeServiceTask(ServiceTask):
    service_version = 'v1.0.20210524'
    service_name = 'captcha1_recognize'
    mock_result = {
        'text': '',
    }
    require_field = {
        "_image_info",
    }
    binding_service = captcha1_recognize
