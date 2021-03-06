from fastapi import APIRouter, Form

from Deployment.ConsumerServices.GeneralService import DownloadImageFromURLServiceTask
from Deployment.ConsumerServices.OCRService import TextDetectServiceTask, TextRecognizeServiceTask, \
    OCRResultVisualizationServiceTask
from Deployment.server_config import IS_MOCK
from Utils.DAG import DAG
from Utils.InterfaceUtils import base_interface_result_decorator, DictInterfaceResult
from Utils.ServiceUtils import wait_and_compose_all_task_result

router = APIRouter()


@router.post('/general_ocr')
@base_interface_result_decorator()
async def general_ocr(
        image_url: str = Form(...),
):
    dag = DAG()
    # 如果mock数据过长，可以写在其他文件，然后这边引用
    mock_result = {
        'bucket_name': 'mock_bucket_name',
        'path': 'mock_path',
        'url': 'mock_url',
    }
    with DictInterfaceResult(mock_result) as to_return_result:
        if IS_MOCK:
            return to_return_result
        download_image_task = DownloadImageFromURLServiceTask(_dag=dag)
        download_image_task.add_dependency_from_value('_image_url', image_url)
        text_detect_task = TextDetectServiceTask(_dag=dag)
        text_detect_task.add_dependency_from_task('_image_info', download_image_task, 'image_info')
        # 由于下载到检测是串行任务，所以这里可以直接await，但是就是需要自己跟上获取detail字段，
        detect_result = (await text_detect_task).service_result
        recognize_task = []
        for i in range(detect_result['box_count']):
            m_recognize_task = TextRecognizeServiceTask(_task_name=f'No. {i} recognize', _dag=dag)
            m_recognize_task.add_dependency_from_task('_image_info', download_image_task, 'image_info')
            m_recognize_task.add_dependency_from_value('_box_info', detect_result['box_info'][i])
            recognize_task.append(m_recognize_task)
        recognize_result = await wait_and_compose_all_task_result(*recognize_task)
        # 上传结果图片，这个环节可以直接在interface中实现，但是为了保证服务的并发量，所以依然是将任务分发出去
        visualization_task = OCRResultVisualizationServiceTask(_dag=dag)
        visualization_task.add_dependency_from_task('_image_info', download_image_task, 'image_info')
        visualization_task.add_dependency_from_value('_box_info_list', detect_result['box_info'])
        visualization_task.add_dependency_from_value('_text_list',
                                                     [m_detail for _, m_detail in recognize_result.items()])
        visualization_task_result = (await visualization_task).service_result
        to_return_result.add_sub_result('bucket_name', visualization_task_result['bucket_name'])
        to_return_result.add_sub_result('path', visualization_task_result['path'])
        to_return_result.add_sub_result('url', visualization_task_result['url'])
        return to_return_result
