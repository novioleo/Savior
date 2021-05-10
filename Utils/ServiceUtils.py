import asyncio
import time
from collections import OrderedDict

from celery import exceptions as celery_exceptions

from Deployment.server_config import SUBTASK_EXECUTE_TIME_LIMIT_SECONDS, TASK_QUEUE
from Utils.DAG import DAG
from Utils.Exceptions import CustomException, \
    RetryExceedLimitException, DAGAbortException
from Utils.misc import get_uuid_name


class ServiceResult:
    def __init__(self, _version, _mock_result):
        self.service_version = _version
        self.start_time = None
        self.cost_time = 0
        self.return_code = 0
        self.return_message = 'success'
        self.service_result = _mock_result.copy()

    def get_result_info(self):
        service_status = {
            'cost_time(seconds)': self.cost_time,
            'return_code': self.return_code,
            'return_message': self.return_message,
            'service_version': self.service_version
        }
        return service_status, self.service_result

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cost_time = int(time.time() - self.start_time)

    def finish(self, _result):
        for m_key, m_value in _result.items():
            self.service_result[m_key] = m_value

    def fail(self, _exception):
        if isinstance(_exception, CustomException):
            self.return_code, self.return_message = _exception.format_exception()
        else:
            self.return_code, self.return_message = -1, str(_exception)


class ServiceTask:
    """
    用于在interface层调用service
    """
    service_version = 'default version'
    service_name = 'default_service'
    mock_result = dict()
    require_field = set()

    binding_service = None

    def __init__(self, _dag: DAG, _count_down=0, _task_name=None, _is_mock=False, _retry_count=5):
        """
        初始化task

        Args:
            _dag:           workflow的dag
            _count_down:    启动时间（秒），启动时间如果非0，则会放入后台等待设定的秒数后进行，不会回传结果。
            _task_name:     任务名称
            _is_mock:       是否是mock状态
        """
        self.filled_field = dict()
        if _task_name is not None:
            self.task_name = _task_name
        else:
            self.task_name = self.service_name
        self.task_id = get_uuid_name()
        self.is_mock = _is_mock
        self.count_down = _count_down
        if self.is_mock:
            self.service_version = 'Mock Version'
        if not self.is_mock:
            assert self.binding_service is not None, f'{self.task_name} not bind to service'
        self.start_time = time.time()
        self.task = None
        self.create_task()
        self.dag = _dag
        self.retry_count = _retry_count

    def create_task(self):
        if self.filled_field.keys() == self.require_field:
            self.task = asyncio.create_task(self.execute())
            self.dag.create_task_node(self)

    def __await__(self):
        assert self.task is not None, 'task not setup success'
        task_result = yield from self.task.__await__()
        if task_result.return_code != 0:
            raise DAGAbortException(
                f'service {self.service_name} cannot go on',
                self.service_name,
                task_result.return_message,
            )
        return task_result

    async def get_request_data(self):
        to_return_request_data = dict()
        all_missed_field = []
        for m_field in self.require_field:
            if m_field not in self.filled_field:
                all_missed_field.append(m_field)
        if len(all_missed_field):
            raise AssertionError(
                f'service {self.service_name} parameter missing:[ {",".join(all_missed_field)} ]')
        all_dependent_task = []
        for m_field_name, m_field_value in self.filled_field.items():
            if isinstance(m_field_value, tuple):
                m_task, m_task_field_name = m_field_value
                all_dependent_task.append((m_task, m_task_field_name, m_field_name))
                self.dag.create_task_dependency(m_task, m_task_field_name, self, m_field_name)
            else:
                to_return_request_data[m_field_name] = m_field_value
                self.dag.create_value_dependency(m_field_value, self, m_field_name)
        if len(all_dependent_task):
            all_dependent_task_results = await asyncio.gather(
                *[_[0] for _ in all_dependent_task]
            )
            for m_result, (m_task, m_task_field_name, m_field_name) in zip(all_dependent_task_results,
                                                                           all_dependent_task):
                m_value = m_result.service_result[m_task_field_name]
                m_value = m_value if not isinstance(m_value, bytes) else m_value.decode('utf-8')
                to_return_request_data[m_field_name] = m_value
        return to_return_request_data

    def add_dependency_from_value(self, _field_name, _field_value):
        assert _field_name in self.require_field, f'{_field_name} DONT NEED in {self.service_name}'
        available_type = [int, float, str, dict, list]
        assert any([isinstance(_field_value, m_type) for m_type in available_type]), \
            f'field value type is {type(_field_value)},which is not support now.'
        self.filled_field[_field_name] = _field_value
        self.create_task()

    def add_dependency_from_task(self, _field_name, _task, _task_field_name):
        assert _field_name in self.require_field, f'{_field_name} DONT NEED in {self.service_name}'
        assert _task_field_name in _task.mock_result, f'Task {_task.service_name} DONT HAVE "{_task_field_name}"'
        self.filled_field[_field_name] = (_task, _task_field_name)
        self.create_task()

    async def execute(self):
        # 如果说有task name则算子的名称按task name来定
        if self.task_name is None:
            self.task_name = self.service_name
        with ServiceResult(self.service_version, self.mock_result) as to_return_result:
            if self.is_mock:
                return to_return_result
            request_data = await self.get_request_data()
            # 获取实际运行启动时间
            self.start_time = time.time()
            try:
                celery_task = self.binding_service.apply_async(
                    kwargs=request_data,
                    countdown=self.count_down,
                    queue=TASK_QUEUE,
                )
                if self.count_down == 0:
                    api_result_dict = celery_task.get(propagate=True, timeout=SUBTASK_EXECUTE_TIME_LIMIT_SECONDS, )
                    self.dag.set_task_node_result(self, api_result_dict)
                    to_return_result.finish(api_result_dict)
            except (celery_exceptions.TimeoutError, celery_exceptions.TimeLimitExceeded) as retry_exception:
                self.retry_count -= 1
                if self.retry_count > 0:
                    return await self.execute()
                else:
                    to_return_result.fail(RetryExceedLimitException(f'{self.service_name} retried exceed limit.'))
            except Exception as e:
                to_return_result.fail(e)
            return to_return_result


async def wait_and_compose_all_task_result(*tasks):
    """
    打包所有task的结果

    Args:
        *tasks: 所有的task

    Returns:    所有task的结果的detail

    """
    to_return_result = OrderedDict()
    all_task_results = await asyncio.gather(*tasks)
    for m_task, m_task_result in zip(tasks, all_task_results):
        to_return_result[m_task.task_name] = m_task_result.service_result
    return to_return_result
