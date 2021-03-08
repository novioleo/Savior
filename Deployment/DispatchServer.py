import asyncio
import time
import traceback as tb
from collections import OrderedDict
from functools import wraps
import uvicorn
from fastapi import FastAPI
import Deployment.APIConsumer as consumer
from Utils.Exceptions import ConsumerAlgorithmQueryException, ConsumerAlgorithmRuntimeException, \
    ConsumerComputeException, ConsumerAlgorithmUncatchException
from Deployment.server_config import DEPLOY_VERSION, SERVER_NAME, SUBTASK_EXECUTE_TIME_LIMIT_SECONDS, \
    AVAILABLE_INTERFACES

app = FastAPI(title=SERVER_NAME, version=DEPLOY_VERSION)
for m_router, m_path_prefix in AVAILABLE_INTERFACES:
    app.include_router(m_router, prefix=m_path_prefix)


async def get_coroutine_task_result(_task):
    if _task.done():
        return _task.result()
    else:
        return await _task


def algorithm_result_compose(
        _version,
        _api_result,
        _time_cost,
):
    to_return_result = OrderedDict()
    to_return_result['version'] = _version
    to_return_result['detail'] = _api_result
    to_return_result['time_cost'] = '%d ms' % (_time_cost * 1000)
    return to_return_result


def common_request_interface(_request_algorithm_name, _target_api_name):
    def _wrap_function(_func):
        @wraps(_func)
        async def _execute(*args, **kwargs):
            # 计算包括依赖在内的总的时间
            start_time = time.time()
            # 如果说有task name则算子的名称按task name来定
            if 'task_name' in kwargs:
                task_name = kwargs['task_name']
                kwargs.pop('task_name')
            else:
                task_name = _target_api_name

            if '_countdown' in kwargs:
                countdown = kwargs['_countdown']
                kwargs.pop('_countdown')
            else:
                countdown = 0
            request_data = await _func(*args, **kwargs)
            # 避免orjson dumps的结果为bytes的情况
            request_data = {m_key: m_value if not isinstance(m_value, bytes) else m_value.decode('utf-8')
                            for m_key, m_value in request_data.items()}
            try:
                task = getattr(consumer, _target_api_name).apply_async(
                    kwargs=request_data,
                    countdown=countdown,
                    queue='request_queue',
                )
                if countdown == 0:
                    api_result_dict = None

                    for i in range(SUBTASK_EXECUTE_TIME_LIMIT_SECONDS):
                        if not task.ready():
                            await asyncio.sleep(1)
                        else:
                            api_result_dict = task.get(propagate=True)
                            break
                    # API服务可能已经没有多余的计算资源了
                    if api_result_dict is None:
                        raise ConsumerAlgorithmQueryException('api consumer没有响应')
                    if api_result_dict['code'] != 0:
                        raise ConsumerAlgorithmRuntimeException(api_result_dict['code'], api_result_dict["msg"])
                    else:
                        return task_name, \
                               api_result_dict['detail'], \
                               api_result_dict['version'], \
                               time.time() - start_time
            except ConsumerComputeException as rare:
                raise rare
            except Exception as e:
                raise ConsumerAlgorithmUncatchException(tb.format_exc())

        return _execute

    return _wrap_function


if __name__ == '__main__':
    from Deployment.server_config import DISPATCH_SERVER_PORT

    uvicorn.run(app, host="0.0.0.0", port=DISPATCH_SERVER_PORT)
