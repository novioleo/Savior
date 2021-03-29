from abc import ABC

import cv2
import numpy as np

from Operators.DummyAlgorithm import DummyAlgorithm


class NRIQA(DummyAlgorithm, ABC):
    name = '无参考IQA'
    __version__ = 'v1.0.20210329'

    def __init__(self, _is_test, _image_size=128):
        super().__init__(_is_test)
        self.image_size = _image_size

    def image_preprocess(self, _image):
        resized_image = cv2.resize(_image, (self.image_size, self.image_size), )
        if len(resized_image.shape) == 3:
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = resized_image
        normalized_image = gray_image / 255
        return normalized_image


class Brenner(NRIQA):
    """
    值越大，越清晰
    """
    name = 'Brenner'
    __version__ = 'v1.0.20210329'

    def __init__(self, _is_test, _image_size=128, _delta_x=2):
        super().__init__(_is_test, _image_size)
        self.delta_x = _delta_x
        assert _delta_x < self.image_size, 'delta_x must small than image size'

    def execute(self, _image):
        candidate_image = self.image_preprocess(_image)
        a = candidate_image[..., :-self.delta_x]
        b = candidate_image[..., self.delta_x:]
        return np.linalg.norm(a - b) / (self.image_size ** 2)


class Laplacian(NRIQA):
    """
    值越大，越清晰
    """
    name = 'Laplacian'
    __version__ = 'v1.0.20210329'

    def execute(self, _image):
        candidate_image = self.image_preprocess(_image)
        return cv2.Laplacian(candidate_image, cv2.CV_64F).var()


if __name__ == '__main__':
    from argparse import ArgumentParser

    ag = ArgumentParser('Image IQA Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
    blurred_img = cv2.GaussianBlur(img, (7, 7), 11)
    brenner_handler = Brenner(True,_delta_x=5)
    laplacian_handler = Laplacian(True)
    for m_handler in [brenner_handler, laplacian_handler]:
        print(m_handler.name, 'clarity image score', m_handler.execute(img))
        print(m_handler.name, 'blurred image score', m_handler.execute(blurred_img))
