from Operators.DummyOperator import DummyOperator


class VideoExtractOperator(DummyOperator):
    name = '视频关键帧提取'
    __version__ = 'v1.0.20210326'

    def __init__(self, _is_test):
        super().__init__(_is_test)

    def execute(self, _video):
        pass
