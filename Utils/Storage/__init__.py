from Deployment.server_config import OSS_TYPE, OSS_INFO
from Utils.Storage.BaseOSS import CloudObjectStorage
from Utils.Storage.Dummy import DummyOSS
from Utils.Storage.Minio import MinioOSS


def get_oss_handler():
    if OSS_TYPE == 'MINIO':
        to_return_handler = MinioOSS(
            _endpoint=OSS_INFO['ENDPOINT'],
            _access_key=OSS_INFO['ACCESS_KEY'],
            _secret_key=OSS_INFO['SECRET_KEY'],
            _region=OSS_INFO['REGION'],
        )
    elif OSS_TYPE == 'DUMMY':
        to_return_handler = DummyOSS(
            None, None, None
        )
    else:
        raise NotImplementedError(f'oss client {OSS_TYPE} not implement')
    return to_return_handler


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    import cv2
    import numpy as np

    ag = ArgumentParser('OSS Test Example')
    ag.add_argument('--end_point_url', required=True, type=str, help='OSS服务器的URL')
    ag.add_argument('--end_point_port', required=True, type=str, help='OSS服务器的port')
    ag.add_argument('--backend_type', required=True, choices=['minio', 'dummy'], type=str, help='OSS服务器的类型')
    ag.add_argument('-a', '--access_key', type=str, required=True, help='access key')
    ag.add_argument('-s', '--secret_key', type=str, required=True, help='secret key')
    ag.add_argument('-r', '--region', type=str, default=None, help='地域')
    ag.add_argument('-b', '--bucket_name', type=str, default='testbucket', help='bucket名称')
    ag.add_argument('-i', '--image_path', type=str, required=True, help='测试用的图像路径')
    args = ag.parse_args()
    if args.backend_type == 'minio':
        oss_operator = MinioOSS(
            f'{args.end_point_url}:{args.end_point_port}',
            args.access_key,
            args.secret_key,
            args.region
        )
    elif args.backend_type == 'dummy':
        oss_operator = DummyOSS(
            None, None, None
        )
    else:
        raise NotImplementedError(f'{args.backend_type} not implement')
    bucket_name = args.bucket_name
    oss_operator.create_bucket(bucket_name)
    img = cv2.imread(args.image_path)
    resize_img = cv2.resize(img, (128, 128))
    upload_image_path = os.path.join('test_images', 'image')
    upload_image_path = oss_operator.upload_image_file(
        bucket_name, upload_image_path, resize_img, True, 90
    )
    image_numpy = oss_operator.download_image_file(bucket_name, upload_image_path)
    print('image_numpy_request_path', oss_operator.get_retrieve_url(bucket_name, upload_image_path))
    cv2.imwrite('resize_image.png', image_numpy)
    test_numpy_array = [
        np.random.randint(3, 5, (3, 3)).astype(np.uint8),
        np.random.uniform(0, 1, (5, 5))
    ]
    upload_test_numpy_path = os.path.join('test_numpy_arrays', 'test_array')
    oss_operator.upload_numpy_array(bucket_name, upload_test_numpy_path, test_numpy_array)
    download_test_numpy_array = oss_operator.download_numpy_array(bucket_name, upload_test_numpy_path)
    for m_array, m_download_array in zip(test_numpy_array, download_test_numpy_array):
        assert np.sum(m_array - m_download_array) == 0, 'there is different'
