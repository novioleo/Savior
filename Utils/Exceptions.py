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


class ImageFileSizeAbnormalException(AlgorithmInputParameterException):
    """
    图像文件大小不正常异常
    """
    MIRROR_CODE = 13


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


class AlgorithmRuntimeException(AlgorithmOperatorException):
    """
    算子运行时错误
    """
    MAJOR_CODE = 2
