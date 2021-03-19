from abc import ABC

from Utils.InferenceHelpers.BaseInferenceHelper import CustomInferenceHelper


class DummyInferenceHelper(CustomInferenceHelper, ABC):
    """

    dummy inference helper 是为了解决在测试过程中不用去考虑网络是否正确，可以方便团队解耦进行开发。

    """

    def __init__(self, _algorithm_name):
        super().__init__()
        self.name = _algorithm_name
        self.type_name = 'dummy'
