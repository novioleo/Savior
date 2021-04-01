import os

from minio import Minio, S3Error

from Utils.Exceptions import ObjectNotFoundOnOSS
from Utils.Storage.BaseOSS import CloudObjectStorage
import datetime


class MinioOSS(CloudObjectStorage):

    def __init__(self, _endpoint, _access_key, _secret_key, _region=None):
        super().__init__(_endpoint, _access_key, _secret_key)
        self.region = _region
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            region=self.region,
            secure=False,
        )

    def create_bucket(self, _bucket_name):
        if not self.client.bucket_exists(_bucket_name):
            self.client.make_bucket(_bucket_name, location=self.region)

    def check_file_exist(self, _bucket_name, _object_path):
        try:
            self.client.stat_object(_bucket_name, _object_path)
        except S3Error as s3e:
            if s3e.code == 'NoSuchKey':
                raise ObjectNotFoundOnOSS(os.path.join(_bucket_name, _object_path) + ' not found')
            elif s3e.code == 'NoSuchBucket':
                raise ObjectNotFoundOnOSS(_bucket_name + ' not found')
            else:
                # 权限deny和bucket不合法的情况直接抛出，如果需要处理，可以自行catch
                raise s3e

    def download_data(self, _bucket_name, _object_path):
        self.check_file_exist(_bucket_name, _object_path)
        response = self.client.get_object(_bucket_name, _object_path)
        data_bytes = response.data
        response.close()
        response.release_conn()
        return data_bytes

    def upload_data(self, _bucket_name, _object_path, _to_upload_object_bytes):
        self.create_bucket(_bucket_name)
        return self.client.put_object(_bucket_name, _object_path, _to_upload_object_bytes, -1,
                                      part_size=5 * 1024 * 1024).object_name

    def get_retrieve_url(self, _bucket_name, _object_path, _expire_seconds=86400 * 7):
        # note: 需要设置bucket的policy，设置为可读，确保url能正常访问
        # url有效期为7天
        url = self.client.presigned_get_object(_bucket_name, _object_path,
                                               expires=datetime.timedelta(seconds=_expire_seconds))
        return url
