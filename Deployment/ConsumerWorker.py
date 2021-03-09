import asyncio
import traceback as tb
from concurrent.futures.thread import ThreadPoolExecutor
from functools import wraps

from celery import Celery
from kombu import Queue

from Deployment.server_config import WORKER_RABBITMQ_USERNAME, WORKER_RABBITMQ_PASSWORD, WORKER_RABBITMQ_HOST, \
    WORKER_RABBITMQ_PORT, WORKER_RABBITMQ_VHOST, IS_MOCK
from Utils.Exceptions import CustomException

celery_worker_app = Celery(
    "algorithm_worker",
    backend="rpc://",
    broker=f"amqp://{WORKER_RABBITMQ_USERNAME}:{WORKER_RABBITMQ_PASSWORD}@"
           f"{WORKER_RABBITMQ_HOST}:{WORKER_RABBITMQ_PORT}/{WORKER_RABBITMQ_VHOST}",
)

celery_worker_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

celery_worker_app.conf.task_queues = (
    Queue('algorithm_request_queue', routing_key='Deployment.APIConsumer.#'),
)

# 用于放置一些非计算密集型任务
background_thread_pool_executor = ThreadPoolExecutor()


def async_task(_async_function):
    @wraps(_async_function)
    def wrapper(*args, **kwargs):
        return asyncio.run(_async_function(*args, **kwargs))

    return wrapper


def common_api_interface(_mock_result, _algorithm_version):
    """
    针对于普遍的api接口的处理逻辑（复杂接口需要自行设计）
    简化程序重复性开发，每个api只需要实现基本的功能

    Args:
        _mock_result:   mock状态下的结果输出
        _algorithm_version:     算法版本
    """

    def _wrap_function(_func):
        @wraps(_func)
        async def _execute(*args, **kwargs):
            code = 0
            msg = 'success'
            if not IS_MOCK:
                try:
                    # 每个算子实现基本功能并返回dict
                    temp_result = await _func(*args, **kwargs)
                    # 避免字段不全的情况，如果字段不全，最多只是返回字段值是错的，不会缺少字段
                    detail_result = _mock_result.copy()
                    detail_result.update(temp_result)
                except CustomException as ce:
                    code, msg = ce.format_exception()
                    detail_result = {}
                except Exception as e:
                    code, msg = 999, tb.format_exc()
                    detail_result = {}
            else:
                detail_result = _mock_result
            return {
                'code': code,
                'msg': msg,
                'version': _algorithm_version if not IS_MOCK else 'Mock Version',
                'detail': detail_result
            }

        return _execute

    return _wrap_function
