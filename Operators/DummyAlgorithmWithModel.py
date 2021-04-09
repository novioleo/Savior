from abc import ABC, abstractmethod

from Operators.DummyAlgorithm import DummyAlgorithm


class DummyAlgorithmWithModel(DummyAlgorithm, ABC):

    def __init__(self, _inference_config, _is_test):
        super().__init__(_is_test)
        self.inference_config = _inference_config
        self.inference_helper = None
        self.get_inference_helper()

    @abstractmethod
    def get_inference_helper(self):
        pass
