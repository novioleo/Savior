from typing import Optional

from fastapi import APIRouter, Form

from Deployment.ConsumerServices.GeneralService import DownloadImageFromURLServiceTask, ParseImageFromBase64ServiceTask
from Deployment.ConsumerServices.RecaptchaService import Captcha1RecognizeServiceTask
from Deployment.server_config import IS_MOCK
from Utils.DAG import DAG
from Utils.Exceptions import InputParameterAbsentException, InputParameterAbnormalException
from Utils.InterfaceUtils import base_interface_result_decorator, DictInterfaceResult

router = APIRouter()


@router.post('/captcha1_recognize')
@base_interface_result_decorator()
async def general_ocr(
        image_url: Optional[str] = Form(None),
        image_base64: Optional[str] = Form(None),
):
    dag = DAG()
    # 如果mock数据过长，可以写在其他文件，然后这边引用
    mock_result = {
        'text': '',
    }
    with DictInterfaceResult(mock_result) as to_return_result:
        if IS_MOCK:
            return to_return_result
        if not (image_url or image_base64):
            raise InputParameterAbsentException('至少有一个参数image_url或者image_base64')
        if image_url and image_base64:
            raise InputParameterAbnormalException('只能有一个image_url或者image_base64')
        if image_url:
            download_image_task = DownloadImageFromURLServiceTask(_dag=dag)
            download_image_task.add_dependency_from_value('_image_url', image_url)
            image_info = (await download_image_task).service_result['image_info']
        elif image_base64:
            decode_image_task = ParseImageFromBase64ServiceTask(_dag=dag)
            decode_image_task.add_dependency_from_value('_base64_string', image_base64)
            image_info = (await decode_image_task).service_result['image_info']
        captcha1_recognize_task = Captcha1RecognizeServiceTask(_dag=dag)
        captcha1_recognize_task.add_dependency_from_value('_image_info', image_info)
        recognize_result = (await captcha1_recognize_task).service_result['text']
        to_return_result.add_sub_result('text', recognize_result)
        return to_return_result
