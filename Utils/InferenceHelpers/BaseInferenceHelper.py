from abc import ABC, abstractmethod
import numpy as np


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


class NCNNInferenceHelper(CustomInferenceHelper, ABC):
    def __init__(self, _algorithm_name):
        self.name = _algorithm_name
        self.type_name = 'ncnn'
        self.handler = None


class TritonInferenceHelper(CustomInferenceHelper, ABC):

    def __init__(self, _algorithm_name):
        self.name = _algorithm_name
        self.type_name = 'triton'
