import os

from qcloud_cos import CosConfig, CosS3Client

from Utils.Exceptions import ObjectNotFoundOnOSS
from Utils.Storage.BaseOSS import CloudObjectStorage


class COSOSS(CloudObjectStorage):

    def __init__(self, _secret_id, _secret_key, _appid, _region):
        super().__init__(None, _secret_id, _secret_key)
        self.region = _region
        self.appid = _appid
        self.config = CosConfig(Region=self.region, SecretId=_secret_id, SecretKey=_secret_key)

    def create_bucket(self, _bucket_name):
        bucket_fullname = _bucket_name + "-" + self.appid
        client = CosS3Client(self.config)
        # 默认创建的是private的bucket，只可通过pre signed url访问
        if not client.bucket_exists(bucket_fullname):
            client.create_bucket(
                Bucket=bucket_fullname,
                ACL='private'
            )

    def check_file_exist(self, _bucket_name, _object_path):
        bucket_fullname = _bucket_name + "-" + self.appid
        # 获取客户端对象
        client = CosS3Client(self.config)
        if not client.object_exists(bucket_fullname, _object_path):
            raise ObjectNotFoundOnOSS(os.path.join(bucket_fullname, _object_path) + ' not found')

    def download_data(self, _bucket_name, _object_path):
        self.check_file_exist(_bucket_name, _object_path)
        bucket_fullname = _bucket_name + "-" + self.appid
        # 获取客户端对象
        client = CosS3Client(self.config)
        response = client.get_object(bucket_fullname, _object_path)
        all_bytes = b''
        while True:
            chunk = response['Body'].read(1024)
            if not chunk:
                break
            all_bytes += chunk
        return all_bytes

    def upload_data(self, _bucket_name, _object_path, _to_upload_object_bytes):
        bucket_fullname = _bucket_name + "-" + self.appid
        # 获取客户端对象
        client = CosS3Client(self.config)
        self.create_bucket(_bucket_name)
        result = client.put_object(bucket_fullname, _to_upload_object_bytes, _object_path)
        return _object_path

    def get_retrieve_url(self, _bucket_name, _object_path, _expire_seconds=86400 * 7):
        bucket_fullname = _bucket_name + "-" + self.appid
        # 获取客户端对象
        client = CosS3Client(self.config)
        # url默认7天过期
        return client.get_presigned_download_url(bucket_fullname, _object_path, Expired=_expire_seconds)
