import functools
import time
import traceback as tb
from abc import abstractmethod

from fastapi.responses import ORJSONResponse

from Utils.Exceptions import DAGAbortException, CustomException


class InterfaceResult:
    def __init__(self, _mock_result):
        self.interface_result = _mock_result
        self.start_time = 0
        self.cost_time = 0
        self.return_code = 0
        self.return_message = 'success'

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cost_time = int(time.time() - self.start_time)

    def fail(self, _code, _message):
        self.interface_result = dict()
        self.return_code = _code
        self.return_message = _message

    @abstractmethod
    def dump2response(self):
        pass


class DictInterfaceResult(InterfaceResult):
    """
    针对于返回详情为
    """

    def __init__(self, _mock_result):
        assert isinstance(_mock_result, dict), 'mock result must be dict'
        super().__init__(_mock_result)

    def add_sub_result(self, _name, _sub_result):
        self.interface_result[_name] = _sub_result

    def dump2response(self):
        return ORJSONResponse({
            'result_detail': self.interface_result,
            'time_cost(seconds)': self.cost_time,
            'code': self.return_code,
            'message': self.return_message,
        })


def base_interface_result_decorator():
    def wrapper(_function):
        @functools.wraps(_function)
        async def wrapped(*args, **kwargs):
            """
            主要是完成将interface运行的时候报的所有错误都catch，保证服务器不会崩

            Returns:    response对象

            """
            interface_result = DictInterfaceResult(dict())
            try:
                interface_result = await _function(*args, **kwargs)
            except DAGAbortException as de:
                # 针对于前置任务没有正确执行的情况
                code, _ = de.format_exception()
                interface_result.fail(code, de.real_reason)
            except CustomException as ce:
                code, message = ce.format_exception()
                interface_result.fail(code, message)
            except Exception as e:
                code, message = -1, tb.format_exc()
                interface_result.fail(code, message)
            return interface_result.dump2response()

        return wrapped

    return wrapper
