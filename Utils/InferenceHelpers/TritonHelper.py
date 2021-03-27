from abc import ABC

import numpy as np
import tritonclient.grpc as grpcclient

from Utils.Exceptions import TritonServerCannotConnectException, TritonServerNotReadyException, \
    InferenceTensorCheckFailException
from Utils.InferenceHelpers.BaseInferenceHelper import CustomInferenceHelper, ImageTensorInfo


class TritonInferenceHelper(CustomInferenceHelper, ABC):
    numpy_data_type_mapper = {
        np.half.__name__: "FP16",
        np.float32.__name__: "FP32",
        np.float64.__name__: "FP64",
        np.bool8.__name__: "BOOL",
        np.uint8.__name__: "UINT8",
        np.int8.__name__: "INT8",
        np.short.__name__: "INT16",
        np.int32.__name__: "INT32",
    }

    def __init__(self, _algorithm_name, _server_url, _server_port, _model_name, _model_version):
        super().__init__()
        self.name = _algorithm_name
        self.type_name = 'triton'
        self.target_url = '%s:%s' % (_server_url, _server_port)
        self.model_name = _model_name
        self.model_version = str(_model_version)

        try:
            # 新版本的triton client的send和receive的length都超过了1GB，足够霍霍了
            # self.triton_client = CustomInferenceServerClient(url=self.target_url)
            self.triton_client = grpcclient.InferenceServerClient(url=self.target_url)
        except Exception as e:
            raise TritonServerCannotConnectException(f'triton server {self.target_url} connect fail')
        if not self.triton_client.is_server_ready():
            raise TritonServerNotReadyException(f'triton server {self.target_url} not ready')

    def add_image_input(self, _input_name, _input_shape, _input_description, _mean_and_std):
        self.all_inputs[_input_name] = ImageTensorInfo(_input_name, _input_shape, _input_description, _mean_and_std)

    def infer(self, _need_tensor_check=False, **_input_tensor):
        inputs = []
        assert _input_tensor.keys() == set(self.all_inputs.keys()), f'{self.model_name} the input tensor not match'
        for m_name, m_tensor_info in self.all_inputs.items():
            m_tensor = _input_tensor[m_name]
            if not (isinstance(m_tensor, np.ndarray) and m_tensor.dtype.name in self.numpy_data_type_mapper):
                raise InferenceTensorCheckFailException(f'tensor {m_name} is available numpy array')
            if _need_tensor_check:
                check_status, check_result = m_tensor_info.tensor_check(m_tensor)
                if not check_status:
                    raise InferenceTensorCheckFailException(check_result)
            m_normalized_tensor = m_tensor_info.normalize(m_tensor, _tensor_format='chw').astype(m_tensor.dtype)
            m_infer_input = grpcclient.InferInput(m_name,
                                                  m_normalized_tensor.shape,
                                                  self.numpy_data_type_mapper[m_normalized_tensor.dtype.name]
                                                  )
            m_infer_input.set_data_from_numpy(m_normalized_tensor)
            inputs.append(m_infer_input)
        results = self.triton_client.infer(model_name=self.model_name,
                                           model_version=self.model_version,
                                           inputs=inputs)
        to_return_result = dict()
        for m_result_name in self.all_outputs.keys():
            to_return_result[m_result_name] = results.as_numpy(m_result_name)
        return to_return_result
