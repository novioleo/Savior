from Operators.DummyAlgorithm import DummyAlgorithm
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import numpy as np
from collections import Counter
from Utils.GeometryUtils import rotate_degree_img, resize_with_long_side
import cv2


def get_max_frequency_element(_elements):
    all_element_frequency = Counter(_elements)
    max_frequency = max(all_element_frequency.values())
    to_return_all_max_frequency_elements = []
    for m_element, m_element_count in all_element_frequency.items():
        if m_element_count == max_frequency:
            to_return_all_max_frequency_elements.append(m_element)
    return to_return_all_max_frequency_elements


def calculate_deviation(_angle):
    angle_in_degrees = np.abs(_angle)
    deviation = np.abs(np.pi / 4 - angle_in_degrees)
    return deviation


class TextDocumentDeskewOperator(DummyAlgorithm):
    """
    只支持小于45度的角度矫正，对于超过45度的，那么就不能保证文字的方向了。

    经过测试，图像在90度和60度左右的时候就会将图像旋转为竖直状态。
    如果要保证图像准确率更高，可以考虑使用一个图片方向的分类器（4分类），然后将大方向正确的图再进行deskew，基本可以解决绝大部分情况。
    如果对图像进行deskew再进行分类也可以，但是可能效果没有前者好。
    """

    name = '文本文档图像方向矫正'
    __version__ = 'v1.0.20210426'

    def __init__(self, _is_test, _sigma=3.0, _num_peaks=20):
        super().__init__(_is_test)
        self.sigma = _sigma
        self.num_peaks = _num_peaks

    def execute(self, _image):
        # 避免计算耗时过长
        resized_img = resize_with_long_side(_image, 1024)
        gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        edges = canny(gray_image, sigma=self.sigma)
        h, a, d = hough_line(edges)
        _, candidate_angle_bins, _ = hough_line_peaks(h, a, d, num_peaks=self.num_peaks)
        if len(candidate_angle_bins) == 0:
            return _image
        absolute_deviations = [calculate_deviation(m_angle) for m_angle in candidate_angle_bins]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        angle_degrees = [np.rad2deg(x) for x in candidate_angle_bins]

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []
        low_bound_of_angle = 44
        high_bound_of_angle = 46
        for m_angle_degree in angle_degrees:
            for m_bin, m_new_angle_degree in zip(
                    [bin_45_90, bin_0_45, bin_0_45n, bin_45_90n],
                    [90 - m_angle_degree, m_angle_degree, -m_angle_degree, 90 + m_angle_degree]
            ):
                deviation_sum = int(m_new_angle_degree + average_deviation)
                if low_bound_of_angle <= deviation_sum <= high_bound_of_angle:
                    m_bin.append(m_angle_degree)

        candidate_angle_bins = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        selected_angle_bin = 0
        selected_angle_bin_length = len(candidate_angle_bins[0])

        for j in range(1, len(candidate_angle_bins)):
            m_len_angles = len(candidate_angle_bins[j])
            if m_len_angles > selected_angle_bin_length:
                selected_angle_bin_length = m_len_angles
                selected_angle_bin = j

        if selected_angle_bin_length:
            candidate_degrees = get_max_frequency_element(candidate_angle_bins[selected_angle_bin])
            mean_degree = np.mean(candidate_degrees)
        else:
            candidate_degrees = get_max_frequency_element(angle_degrees)
            mean_degree = np.mean(candidate_degrees)
        target_to_rotate_angle = mean_degree
        if 0 <= mean_degree <= 90:
            target_to_rotate_angle = mean_degree - 90
        if -90 <= mean_degree < 0:
            target_to_rotate_angle = 90 + mean_degree

        rotated_image, _ = rotate_degree_img(_image, -target_to_rotate_angle)
        return rotated_image


if __name__ == '__main__':
    from argparse import ArgumentParser

    ag = ArgumentParser('Text Document Deskew Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='图片路径')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
    text_document_deskew_handler = TextDocumentDeskewOperator(True)
    deskewed_img = text_document_deskew_handler.execute(img)
    cv2.imwrite('deskewed_img.png', deskewed_img)
