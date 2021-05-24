from Deployment.ConsumerWorker import celery_worker_app
from Deployment.server_config import IS_TEST
from Operators.ExampleDownloadOperator import ImageDownloadOperator
from Operators.ExampleImageStringParseOperator import ImageParseFromBase64
from Utils.ServiceUtils import ServiceTask
from Utils.Storage import get_oss_handler

image_download_op = ImageDownloadOperator(IS_TEST)
base64_image_decode_op = ImageParseFromBase64(IS_TEST)


@celery_worker_app.task(name="ConsumerServices.GeneralService.download_image_from_url")
def download_image_from_url(_image_url):
    oss_helper = get_oss_handler()
    download_result = image_download_op.execute(_image_url, oss_helper, _image_size_threshold=None)
    return {
        'image_info': {
            'bucket_name': download_result['bucket_name'],
            'path': download_result['saved_path'],
            'height': download_result['image_height'],
            'width': download_result['image_width'],
            'channel': download_result['image_channel'],
        },
    }


class DownloadImageFromURLServiceTask(ServiceTask):
    service_version = 'v1.0.20210317'
    service_name = 'download_image_from_url'
    mock_result = {
        'image_info': {
            'bucket_name': '',
            'path': '',
            'height': 0,
            'width': 0,
            'channel': 1,
        },
    }
    require_field = {
        "_image_url",
    }
    binding_service = download_image_from_url


@celery_worker_app.task(name="ConsumerServices.GeneralService.parse_image_from_base64")
def parse_image_from_base64(_base64_string):
    oss_helper = get_oss_handler()
    decode_result = base64_image_decode_op.execute(_base64_string, oss_helper)
    return {
        'image_info': {
            'bucket_name': decode_result['bucket_name'],
            'path': decode_result['saved_path'],
            'height': decode_result['image_height'],
            'width': decode_result['image_width'],
            'channel': decode_result['image_channel'],
        },
    }


class ParseImageFromBase64ServiceTask(ServiceTask):
    service_version = 'v1.0.20210524'
    service_name = 'parse_image_from_base64'
    mock_result = {
        'image_info': {
            'bucket_name': '',
            'path': '',
            'height': 0,
            'width': 0,
            'channel': 1,
        },
    }
    require_field = {
        "_base64_string",
    }
    binding_service = parse_image_from_base64
