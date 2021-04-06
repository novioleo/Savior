# todo: 后续需要重新规划major code和mirror code，前期先这样写
class CustomException(Exception):
    MAJOR_CODE = 0
    MIRROR_CODE = 0

    def format_exception(self):
        return '%d%02d' % (self.MAJOR_CODE, self.MIRROR_CODE), str(self)


class AlgorithmOperatorException(CustomException):
    """
    算子异常基类
    """
    MAJOR_CODE = 1


class AlgorithmInputParameterException(AlgorithmOperatorException):
    """
    算子入参异常
    """
    MIRROR_CODE = 11


class InputParameterAbnormalException(AlgorithmInputParameterException):
    """
    输入参数不合理
    """
    MIRROR_CODE = 14


class InputParameterAbsentException(AlgorithmInputParameterException):
    """
    输入参数缺失
    """
    MIRROR_CODE = 15


class AlgorithmRuntimeException(CustomException):
    """
    算子运行时错误
    """
    MAJOR_CODE = 2


class NetworkInputParameterException(AlgorithmRuntimeException):
    """
    网络入参异常
    """
    MIRROR_CODE = 11


class NetworkInitFailException(AlgorithmRuntimeException):
    """
    网络初始化异常
    """
    MIRROR_CODE = 21


class ConsumerComputeException(CustomException):
    """
    消费者端出现的错误
    """
    MAJOR_CODE = 3


class ConsumerAlgorithmQueryException(ConsumerComputeException):
    """
    请求消费者端结果异常
    """
    MIRROR_CODE = 11


class ConsumerAlgorithmRuntimeException(ConsumerComputeException):
    """
    消费者端计算错误
    """
    MIRROR_CODE = 12


class ConsumerAlgorithmTimeoutException(ConsumerComputeException):
    """
    消费者端计算超时
    """
    MIRROR_CODE = 13


class ConsumerAlgorithmUncatchException(ConsumerComputeException):
    """
    消费者端无法捕捉的错误
    """
    MIRROR_CODE = 99


class DeepLearningInferenceException(CustomException):
    """
    深度学习推理的异常
    """
    MAJOR_CODE = 3


class TritonUncatchException(DeepLearningInferenceException):
    """
    Triton推理无法捕捉的异常
    """
    MIRROR_CODE = 1


class NCNNUncatchaException(DeepLearningInferenceException):
    """
    NCNN推理无法捕捉的异常
    """
    MIRROR_CODE = 2


class InferenceTensorCheckFailException(DeepLearningInferenceException):
    """
    推理时的tensor有效性检查失败
    """
    MIRROR_CODE = 3


class TritonServerCannotConnectException(DeepLearningInferenceException):
    """
    无法连接Triton服务器
    """
    MIRROR_CODE = 11


class TritonServerNotReadyException(DeepLearningInferenceException):
    """
    Triton服务器还未准备好，可能在启动中
    """
    MIRROR_CODE = 12


class TritonModelNotReadyException(DeepLearningInferenceException):
    """
    Triton上面的特定模型还未加载完成
    """
    MIRROR_CODE = 13


class GeneralException(CustomException):
    MAJOR_CODE = 4


class ImageFileSizeAbnormalException(GeneralException):
    """
    图像文件大小不正常
    """
    MIRROR_CODE = 1


class ObjectNotFoundOnOSS(GeneralException):
    """
    OSS上未找到指定目标
    """
    MIRROR_CODE = 2


class ImageDownloadTimeoutException(GeneralException):
    """
    图像下载超时
    """
    MIRROR_CODE = 3


class ImageFormatNotSupportException(GeneralException):
    """
    图像格式不支持
    """
    MIRROR_CODE = 4


class DownloadURLNotAvailableException(GeneralException):
    """
    下载链接不可用
    """
    MIRROR_CODE = 5


class DownloadURLTimeoutException(GeneralException):
    """
    下载链接超时
    """
    MIRROR_CODE = 6


class ImageClassNotSupportToEncode(GeneralException):
    """
    OSS在进行图像编码的时候，格式不支持
    """
    MIRROR_CODE = 7


class VideoExtractMethodNotSupport(GeneralException):
    """
    视频帧提取的方法不支持
    """
    MIRROR_CODE = 8
