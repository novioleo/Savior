import asyncio
import time
from collections import OrderedDict
from celery import exceptions as celery_exceptions
import traceback as tb
from Utils.DAG import DAG
from Deployment.server_config import SUBTASK_EXECUTE_TIME_LIMIT_SECONDS, TASK_QUEUE
from Utils.Exceptions import ConsumerAlgorithmTimeoutException, ConsumerAlgorithmUncatchException
from Utils.misc import get_uuid_name


class ServiceTask:
    """
    用于在interface层调用service
    """
    service_version = 'default version'
    service_name = 'default_service'
    mock_result = dict()
    require_field = set()

    binding_service = None

    def __init__(self, _count_down=0, _task_name=None, _is_mock=False, _dag: DAG = None):
        """
        初始化task

        Args:
            _count_down:    启动时间（秒），启动时间如果非0，则会放入后台等待设定的秒数后进行，不会回传结果。
            _task_name:     任务名称
            _is_mock:       是否是mock状态
            _dag:           workflow的dag
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
        assert self.binding_service is not None, f'{self.task_name} not bind to service'
        self.start_time = time.time()
        self.task = None
        self.create_task()
        self.dag = _dag

    def create_task(self):
        if self.filled_field.keys() == self.require_field:
            self.task = asyncio.create_task(self.execute())
            if self.dag is not None:
                self.dag.create_task_node(self)

    def __await__(self):
        assert self.task is not None, 'task not setup success'
        return self.task.__await__()

    def _decorate_result(self, _result_dict, _time_cost):
        to_return_decorated_result = OrderedDict()
        to_return_decorated_result['version'] = self.service_version
        # 记录具体启动时间
        to_return_decorated_result['start_time'] = int(self.start_time * 1000)
        to_return_decorated_result['detail'] = self.mock_result.copy()
        to_return_decorated_result['time_cost'] = '%0.4f ms' % (_time_cost * 1000)
        for m_key, m_value in _result_dict.items():
            to_return_decorated_result['detail'][m_key] = m_value
        return to_return_decorated_result

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
                if self.dag is not None:
                    self.dag.create_task_dependency(m_task, m_task_field_name, self, m_field_name)
            else:
                to_return_request_data[m_field_name] = m_field_value
                if self.dag is not None:
                    self.dag.create_value_dependency(m_field_value, self, m_field_name)
        if len(all_dependent_task):
            all_dependent_task_results = await asyncio.gather(
                *[_[0] for _ in all_dependent_task]
            )
            for m_result, (m_task, m_task_field_name, m_field_name) in zip(all_dependent_task_results,
                                                                           all_dependent_task):
                m_value = m_result['detail'][m_task_field_name]
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
        # 计算包括依赖在内的总的时间
        start_time = time.time()
        # 如果说有task name则算子的名称按task name来定
        if self.task_name is None:
            self.task_name = self.service_name
        if self.is_mock:
            return self._decorate_result(self.mock_result, time.time() - start_time)
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
                if self.dag is not None:
                    self.dag.set_task_node_result(self, api_result_dict)
                return self._decorate_result(api_result_dict, time.time() - start_time)
        except celery_exceptions.TimeoutError as te:
            raise ConsumerAlgorithmTimeoutException(self.service_name + ' timeout')
        except Exception as e:
            raise ConsumerAlgorithmUncatchException(tb.format_exc())


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
        to_return_result[m_task.task_name] = m_task_result['detail']
    return to_return_result
