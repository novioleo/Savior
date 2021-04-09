from abc import ABC, abstractmethod
from collections import OrderedDict

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


class ImageTensorInfo(TensorInfo):
    def __init__(self, _name, _shape, _description, _mean_and_std=None):
        super().__init__(_name, _shape, _description)
        self.mean_and_std = _mean_and_std
        if _mean_and_std is not None:
            mean, std = self.mean_and_std
            # 针对于灰度和非灰度的图像两种情况
            if len(self.shape) == 2:
                assert len(mean) == 1 and len(std) == 1, \
                    'mean and std are not suit for image tensor'
            elif len(self.shape) == 3:
                assert len(mean) == self.shape[-1] and len(std) == self.shape[-1], \
                    'mean and std are not suit for image tensor'

    def normalize(self, _image, _tensor_format='chw'):
        if self.mean_and_std is not None:
            mean, std = self.mean_and_std
            normalized_tensor = (_image - mean) / std
        else:
            normalized_tensor = _image
        if _tensor_format == 'chw':
            return np.transpose(normalized_tensor, (2, 0, 1))[None, ...]
        elif _tensor_format == 'hwc':
            return _image
        else:
            raise NotImplementedError(f'tensor format {_tensor_format} not implement')


class CustomInferenceHelper(ABC):
    name = 'default'
    type_name = 'default'

    def __init__(self):
        self.all_inputs = OrderedDict()
        self.all_outputs = OrderedDict()

    @abstractmethod
    def check_ready(self):
        pass

    def add_input(self, _input_name, _input_shape, _input_description):
        self.all_inputs[_input_name] = TensorInfo(_input_name, _input_shape, _input_description)

    def add_output(self, _output_name, _output_shape, _output_description):
        self.all_outputs[_output_name] = TensorInfo(_output_name, _output_shape, _output_description)

    def network_input_description(self):
        to_return_description = [
            f'input nums:{len(self.all_inputs)}',
        ]
        for m_input_name, m_input in self.all_inputs.items():
            to_return_description.append(
                f'name:{m_input.name}'
            )
            to_return_description.append(
                f'\tshape:{m_input.shape}'
            )
            to_return_description.append(
                f'\tdescription:{m_input.description}'
            )
            if hasattr(m_input, 'mean_and_std'):
                to_return_description.append(
                    f'\tmean:[{",".join(["%0.5f" % _ for _ in m_input.mean_and_std[0]])}]'
                )
                to_return_description.append(
                    f'\tstd:[{",".join(["%0.5f" % _ for _ in m_input.mean_and_std[1]])}]'
                )

        return '\n'.join(to_return_description)

    def network_output_description(self):
        to_return_description = [
            f'ouput nums:{len(self.all_inputs)}',
        ]
        for m_output_name, m_output in self.all_outputs.items():
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
