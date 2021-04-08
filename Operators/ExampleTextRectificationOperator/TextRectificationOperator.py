from abc import ABC

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel


class TextRectificationOperator(DummyAlgorithmWithModel, ABC):
    """
    可以根据自己的需求定制自己的文本矫正算法
    """
    name = "文本矫正"
    __version__ = 'v1.0.20210408'

    def __init__(self, _inference_config, _is_test):
        super().__init__(_inference_config, _is_test)
