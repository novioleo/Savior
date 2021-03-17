from Deployment.ConsumerWorker import celery_worker_app
from Deployment.server_config import IS_TEST
from Operators.ExampleImageDownloadOperator.ImageDownloadOperator import ImageDownloadOperator
from Utils.ServiceUtils import ServiceTask
from Utils.Storage import get_oss_handler

image_download_op = ImageDownloadOperator(IS_TEST)


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
        },
    }
    require_field = {
        "_image_url",
    }
    binding_service = download_image_from_url
