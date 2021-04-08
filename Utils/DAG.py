from collections import defaultdict

from Utils.Exceptions import PreviousTaskNotFinishException
from Utils.misc import get_date_string, get_uuid_name


class DAG:
    def __init__(self, ):
        self.request_id = f'{get_date_string()}_{get_uuid_name()}'
        self.task_nodes = dict()
        self.dependency = defaultdict(list)
        self.result = dict()

    def dump(self):
        return {
            'request_id': self.request_id,
            'task_nodes': self.task_nodes,
            'dependency': self.dependency,
            'result': self.result
        }

    def create_task_node(self, _task):
        if _task.task_id not in self.task_nodes:
            self.task_nodes[_task.task_id] = _task.__class__.__name__

    def create_task_dependency(self, _source_task, _source_field, _target_task, _target_field):
        self.create_task_node(_source_task)
        self.create_task_node(_target_task)
        self.dependency[_target_task.task_id].append((_target_field, _source_task.task_id, _source_field))

    def create_value_dependency(self, _source_value, _target_task, _target_field):
        self.create_task_node(_target_task)
        self.dependency[_target_task.task_id].append((_target_field, None, _source_value))

    def set_task_node_result(self, _task, _result):
        self.result[_task.task_id] = _result

    async def replay_task(self, _task, *_task_args, **_task_kwargs):
        task_instance = type(_task)(*_task_args, **_task_kwargs)
        for (m_dependency_field, m_source_task, m_source_field) in self.dependency[_task.task_id]:
            if m_source_task is not None:
                if m_source_task in self.result:
                    task_instance.add_dependency_from_value(m_dependency_field, self.result[m_source_field])
                else:
                    raise PreviousTaskNotFinishException(f'task {self.task_nodes[m_source_task]} not finish')
            else:
                task_instance.add_dependency_from_value(m_dependency_field, m_source_field)
        return await task_instance
