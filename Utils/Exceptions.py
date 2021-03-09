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


class ImageFormatNotSupportException(AlgorithmInputParameterException):
    """
    图像格式不支持
    """
    MIRROR_CODE = 12


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


class ConsumerAlgorithmUncatchException(ConsumerComputeException):
    """
    消费者端计算错误
    """
    MIRROR_CODE = 99


class GeneralException(CustomException):
    MAJOR_CODE = 3


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
