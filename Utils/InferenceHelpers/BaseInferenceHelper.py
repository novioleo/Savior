from abc import ABC, abstractmethod

import grpc
import numpy as np
import tritongrpcclient
from tritongrpcclient import grpc_service_pb2_grpc

from Utils.Exceptions import TritonServerCannotConnectException, TritonServerNotReadyException


class TensorInfo:
    def __init__(self, _name, _shape, _description):
        self.name = _name
        self.shape = _shape
        self.description = _description

    def tensor_check(self, _to_input_tensor, _limit_check=None):
        try:
            assert isinstance(_to_input_tensor, np.ndarray), '数据必须为numpy数组'
            if self.shape is not None:
                assert self.shape == _to_input_tensor.shape, '尺寸与预期不符'
            if _limit_check is not None:
                assert _to_input_tensor.nbytes <= _limit_check, '数据过大'
            return True, 'match'
        except AssertionError as ae:
            return False, str(ae)


class CustomInferenceHelper(ABC):
    name = 'default'
    type_name = 'default'
    all_inputs = list()
    all_outputs = list()

    def add_input(self, _input_name, _input_shape, _input_description):
        self.all_inputs.append(TensorInfo(_input_name, _input_shape, _input_description))

    def add_output(self, _output_name, _output_shape, _output_description):
        self.all_outputs.append(TensorInfo(_output_name, _output_shape, _output_description))

    def network_input_description(self):
        to_return_description = [
            f'input nums:{len(self.all_inputs)}',
        ]
        for m_input in self.all_inputs:
            to_return_description.append(
                f'name:{m_input.name}'
            )
            to_return_description.append(
                f'\tshape:{m_input.shape}'
            )
            to_return_description.append(
                f'\tdescription:{m_input.description}'
            )
        return '\n'.join(to_return_description)

    def network_output_description(self):
        to_return_description = [
            f'ouput nums:{len(self.all_inputs)}',
        ]
        for m_output in self.all_outputs:
            to_return_description.append(
                f'name:{m_output.name}'
            )
            to_return_description.append(
                f'\tshape:{m_output.shape}'
            )
            to_return_description.append(
                f'\tdescription:{m_output.description}'
            )
        return '\n'.join(to_return_description)

    def __repr__(self):
        print(
            f"algorithm {self.name} use {self.type_name} for infer\n"
            f"there is the input description:\n{self.network_input_description()}\n"
            f"there is the output description:\n{self.network_output_description()}"
        )

    @abstractmethod
    def infer(self, *args, **kwargs):
        pass


class DummyInferenceHelper(CustomInferenceHelper, ABC):
    """

    dummy inference helper 是为了解决在测试过程中不用去考虑网络是否正确，可以方便团队解耦进行开发。

    """

    def __init__(self, _algorithm_name):
        self.name = _algorithm_name
        self.type_name = 'dummy'


class NCNNInferenceHelper(CustomInferenceHelper, ABC):
    def __init__(self, _algorithm_name):
        self.name = _algorithm_name
        self.type_name = 'ncnn'
        self.handler = None


class CustomInferenceServerClient(tritongrpcclient.InferenceServerClient):
    def __init__(self, url, verbose=False):
        super(CustomInferenceServerClient, self).__init__(url, verbose=False)
        channel_opt = [('grpc.max_send_message_length', 10 * 3 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 5 * 1024 * 1024)]
        self._channel = grpc.insecure_channel(url, options=channel_opt)
        self._client_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(
            self._channel)
        self._verbose = verbose
        self._stream = None


class TritonInferenceHelper(CustomInferenceHelper, ABC):
    numpy_data_type_mapper = {
        np.half.__name__: "FP16",
        np.float32.__name__: "FP32",
        np.float64.__name__: "FP64",
        np.bool.__name__: "BOOL",
        np.uint8.__name__: "UINT8",
        np.int8.__name__: "INT8",
        np.short.__name__: "INT16",
        np.int.__name__: "INT32",
    }

    def __init__(self,
                 _algorithm_name,
                 _server_url, _server_port,
                 _model_name, _model_version,
                 _input_names, _output_names,
                 ):
        self.name = _algorithm_name
        self.type_name = 'triton'
        self.target_url = '%s:%s' % (_server_url, _server_port)
        self.model_name = _model_name
        self.model_version = str(_model_version)
        self.input_names = _input_names
        self.output_names = _output_names

        try:
            self.triton_client = CustomInferenceServerClient(url=self.target_url)
        except Exception as e:
            raise TritonServerCannotConnectException(f'triton server {self.target_url} connect fail')
        if not self.triton_client.is_server_ready():
            raise TritonServerNotReadyException(f'triton server {self.target_url} not ready')

    def infer_request(self, **_input_tensor):
        inputs = []
        assert _input_tensor.keys() == set(self.input_names), f'{self.model_name} the input tensor not match'
        for m_name in self.input_names:
            m_tensor = _input_tensor[m_name]
            assert isinstance(m_tensor, np.ndarray) and \
                   m_tensor.dtype.name in self.numpy_data_type_mapper, \
                f'input tensor [{m_name}] is out of line '
            m_infer_input = tritongrpcclient.InferInput(m_name,
                                                        m_tensor.shape,
                                                        self.numpy_data_type_mapper[m_tensor.dtype.name]
                                                        )
            m_infer_input.set_data_from_numpy(m_tensor)
            inputs.append(m_infer_input)
        results = self.triton_client.infer(model_name=self.model_name,
                                           model_version=self.model_version,
                                           inputs=inputs)
        to_return_result = dict()
        for m_result_name in self.output_names:
            to_return_result[m_result_name] = results.as_numpy(m_result_name)
        return to_return_result
