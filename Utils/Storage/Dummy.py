from Utils.Exceptions import ObjectNotFoundOnOSS
from Utils.Storage.BaseOSS import CloudObjectStorage

import os


class DummyOSS(CloudObjectStorage):
    """

    单机测试用oss，全程存储在本地

    """

    def __init__(self, _endpoint, _access_key, _secret_key):
        super().__init__(_endpoint, _access_key, _secret_key)
        self.temp_directory_path = '/tmp/DummyOSS-temp-directory'
        os.makedirs(self.temp_directory_path, exist_ok=True)

    def create_bucket(self, _bucket_name):
        os.makedirs(os.path.join(self.temp_directory_path, _bucket_name), exist_ok=True)

    def download_data(self, _bucket_name, _object_path):
        target_file_path = os.path.join(self.temp_directory_path, _bucket_name, _object_path)
        if not os.path.exists(target_file_path):
            raise ObjectNotFoundOnOSS(target_file_path + ' not found')
        with open(target_file_path, mode='rb') as to_read:
            return to_read.read()

    def upload_data(self, _bucket_name, _object_path, _to_upload_object_bytes):
        self.create_bucket(_bucket_name)
        target_file_path = os.path.join(self.temp_directory_path, _bucket_name, _object_path)
        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
        with open(target_file_path, mode='wb') as to_write:
            to_write.write(_to_upload_object_bytes.read())
        return _object_path

    def get_retrieve_url(self, _bucket_name, _object_path):
        return os.path.join(self.temp_directory_path, _bucket_name, _object_path)
