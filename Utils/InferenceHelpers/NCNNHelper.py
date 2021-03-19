from abc import ABC

from Utils.InferenceHelpers.BaseInferenceHelper import CustomInferenceHelper


class NCNNInferenceHelper(CustomInferenceHelper, ABC):
    def __init__(self, _algorithm_name):
        super().__init__()
        self.name = _algorithm_name
        self.type_name = 'ncnn'
        self.handler = None
