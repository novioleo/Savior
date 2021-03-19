"""
所有的非算法类原子操作都需要基于DummyOperator进行实现
"""
import logging
import os
from abc import ABC, abstractmethod
from logging import handlers

from Utils.misc import get_date_string


class DummyOperator(ABC):
    # 原子操作名称
    name = 'dummy operator'
    # 版本名称
    __version__ = 'v1.0.20210311'

    def __init__(self, _is_test):
        self.logger = logging.getLogger(f'operator [{self.name}]')
        if _is_test:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        # 会在工作目录下面创建文件夹logs，用于装载日志
        os.makedirs('logs', exist_ok=True)
        date_string = get_date_string()
        log_format = f'%(asctime)s : %(levelname)s : %(process)d: %(thread)x: {self.name}: line %(lineno)d: %(message)s'
        # 每天存储一个文件，存储31天，且文件名包含启动日期
        log_file_handler = handlers.TimedRotatingFileHandler(f'logs/{self.name}_{date_string}启动.log', encoding='utf-8',
                                                             when='D', interval=1, backupCount=31)

        log_file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(log_file_handler)
        self.is_test = _is_test

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        根据自己的需要进行实现一个原子操作

        :param args:    按顺序的参数
        :param kwargs:  关键字参数
        :return:    根据接口文档定义输出的OrderedDict
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)
