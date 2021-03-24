import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import colorsys
from Utils.GeometryUtils import compute_two_points_angle, get_coordinates_of_rotated_box

current_directory = os.path.dirname(__file__)
candidate_font = '田氏颜体大字库2.0.ttf'
annotate_font = ImageFont.truetype(os.path.join(current_directory, candidate_font), size=21)


def generate_colors(_color_num):
    """
    生成一定数量的颜色

    Args:
        _color_num: 颜色的数量

    Returns:    所有生成的颜色

    """
    to_return_palette = []
    # 最多50种颜色
    for i in range(_color_num):
        hue_value = (i % 50) * 0.02
        (r, g, b) = colorsys.hsv_to_rgb(hue_value, 1, 1)
        to_return_palette.append([int(b * 255), int(g * 255), int(r * 255)])
    return to_return_palette


def annotation_multi_horizon_line_on_image(_img, _y_list, _line_color, _line_thickness=4):
    to_return_img = _img.copy()
    for m_y in _y_list:
        cv2.line(to_return_img, (0, m_y), (_img.shape[0] - 1, m_y), _line_color, thickness=_line_thickness)
    return cv2.addWeighted(to_return_img, 0.5, _img, 0.5, 0)


def annotation_horizon_line_on_image(_img, _y, _line_color, _line_thickness=4):
    return annotation_multi_horizon_line_on_image(_img, [_y], _line_color, _line_thickness)


def annotate_bounding_box_on_image(_img, _boxes, _specific_color, _with_index=False, _thickness=4):
    to_return_img = _img.copy()
    if len(_boxes) > 0:
        for m_box_index, m_box in enumerate(_boxes):
            cv2.rectangle(to_return_img, (m_box[0], m_box[1]), (m_box[2], m_box[3]), _specific_color,
                          thickness=_thickness)
            if _with_index:
                cv2.putText(to_return_img,
                            f'{m_box_index}',
                            (m_box[0] + 5, m_box[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            _specific_color
                            )
    return to_return_img


def annotate_circle_on_image(_to_annotate_image, _points, _specific_color, _radius=8, _thickness=2):
    """
    在图中标注多个圆

    Args:
        _to_annotate_image:     待标注图像
        _points:    待标注圆的中心（同opencv配置）
        _specific_color:        圆的颜色（同opencv配置）
        _radius:    圆的半径（同opencv配置）
        _thickness: 圆的厚度（同opencv配置）

    """
    h, w = _to_annotate_image.shape[:2]
    if len(_points) > 0:
        for m_point in _points:
            cv2.circle(_to_annotate_image, (int(m_point[0] * w), int(m_point[1] * h)), _radius, _specific_color,
                       thickness=_thickness)


def annotate_polygon_on_image(_img, _polygon, _specific_color, _with_index=False, _is_transparent=True):
    to_return_img = _img.copy()
    if isinstance(_polygon, list):
        _polygon = np.array(_polygon, dtype=np.int)
    cv2.fillPoly(to_return_img, [_polygon, ], _specific_color)
    if _is_transparent:
        to_return_img = cv2.addWeighted(to_return_img, 0.5, _img, 0.5, 0)
    return to_return_img


def __annotation_text_on_image(_img, _text_start_position, _text_color, _text):
    img_pil = Image.fromarray(_img)
    to_draw_image = ImageDraw.Draw(img_pil)
    to_draw_image.multiline_text(_text_start_position, _text, fill=_text_color, font=annotate_font)
    to_return_img = np.asarray(img_pil)
    return to_return_img


def annotation_angle_on_image(_img, _start_point, _middle_point, _end_point, _line_color, _text_color, _angle):
    """
    在图上画一个角

    :param _img:    需要标注的图
    :param _start_point:    起点（顺时针）
    :param _middle_point:   中点
    :param _end_point:  终点（顺时针）
    :param _line_color:     线条颜色
    :param _text_color:     文本颜色
    :param _angle:   当前角度
    :return:
    """
    to_return_img = _img.copy()
    cv2.line(to_return_img, (_start_point[0], _start_point[1]), (_middle_point[0], _middle_point[1]), _line_color, 2)
    cv2.line(to_return_img, (_middle_point[0], _middle_point[1]), (_end_point[0], _end_point[1]), _line_color, 2)
    cv2.circle(to_return_img, (_middle_point[0], _middle_point[1]), 3, (0, 255, 0), 3)
    angle_1 = compute_two_points_angle(_middle_point, _start_point)
    angle_2 = compute_two_points_angle(_middle_point, _end_point)
    start_angle = 0
    if angle_2 < angle_1:
        angle_2 = angle_2 + 360 - angle_1
        start_angle = angle_1
        angle_1 = 0
    cv2.ellipse(to_return_img, (_middle_point[0], _middle_point[1]), (15, 15), start_angle, angle_1, angle_2,
                _line_color, 2)
    to_return_img = __annotation_text_on_image(to_return_img, (_middle_point[0] + 5, _middle_point[1] + 5),
                                               _text_color, str(_angle))
    return to_return_img


def annotation_multi_horizon_width(_img, _y, _x_list, _line_color, _text_color, _text_list,
                                   _thickness=1,
                                   _with_arrow=True):
    """
    横向标注多个宽度

    :param _img:    需要标注的图像
    :param _y:  当前直线所在高度
    :param _x_list: 所有x的列表
    :param _line_color:     线条颜色（bgr）
    :param _text_color:     文本颜色（bgr）
    :param _text_list:  每个区间需要显示的文本
    :param _thickness: 线条粗细
    :param _with_arrow: 线条两端是否带箭头
    :return:    标注后的图像
    """
    assert len(_x_list) - 1 == len(_text_list), '线段数与字符串数不匹配'
    to_return_img = _img.copy()
    # 需要绘制：
    # 1. 双向箭头线
    # 2. 箭头到头的直线
    # 3. 线条对应的文字
    for m_index, (m_start_x, m_end_x, m_text) in enumerate(zip(_x_list[:-1], _x_list[1:], _text_list)):
        if _with_arrow:
            cv2.arrowedLine(to_return_img, (m_start_x, _y), (m_end_x, _y), _line_color, thickness=_thickness)
            cv2.arrowedLine(to_return_img, (m_end_x, _y), (m_start_x, _y), _line_color, thickness=_thickness)
        else:
            cv2.line(to_return_img, (m_start_x, _y), (m_end_x, _y), _line_color, thickness=_thickness)
            cv2.line(to_return_img, (m_end_x, _y), (m_start_x, _y), _line_color, thickness=_thickness)
        # 文本在最左侧
        text_start_x = m_start_x
        text_start_y = _y + (10 if m_index % 2 == 0 else -annotate_font.size - 10)
        to_return_img = __annotation_text_on_image(to_return_img, (text_start_x, text_start_y), _text_color, m_text)
    for m_x in _x_list:
        cv2.line(to_return_img, (m_x, _y - 12), (m_x, _y + 12), _line_color, thickness=_thickness)
    return to_return_img


def annotation_horizon_width(_img, _y, _start_x, _end_x, _line_color, _text_color, _text):
    """
    横向标注宽度

    :param _img:    需要标注的图像
    :param _y:  当前直线所在高度
    :param _start_x:    起始x
    :param _end_x:  结束x
    :param _line_color:     线条颜色（bgr）
    :param _text_color:     文本颜色（bgr）
    :param _text:  需要显示的文本
    :return:    标注后的图像
    """
    return annotation_multi_horizon_width(_img, _y, [_start_x, _end_x], _line_color, _text_color, [_text])


def annotation_multi_vertical_height(_img, _x, _y_list, _line_color, _text_color, _text_list,
                                     _thickness=1,
                                     _with_arrow=True):
    """
    纵向标注多个高度

    :param _img:    需要标注的图像
    :param _x:  当前直线所在宽度
    :param _y_list:  所有y的列表
    :param _line_color:     线条颜色（bgr）
    :param _text_color:     文本颜色（bgr）
    :param _text_list:  所有需要显示的文本
    :param _thickness: 线条粗细
    :param _with_arrow: 线条两端是否带箭头
    :return:    标注后的图像
    """
    assert len(_y_list) - 1 == len(_text_list), '线段数与字符串数不匹配'
    to_return_img = _img.copy()
    # 需要绘制：
    # 1. 双向箭头线
    # 2. 箭头到头的直线
    # 3. 线条对应的文字
    for m_start_y, m_end_y, m_text in zip(_y_list[:-1], _y_list[1:], _text_list):
        if _with_arrow:
            cv2.arrowedLine(to_return_img, (_x, m_start_y), (_x, m_end_y), _line_color, thickness=_thickness)
            cv2.arrowedLine(to_return_img, (_x, m_end_y), (_x, m_start_y), _line_color, thickness=_thickness)
        else:
            cv2.line(to_return_img, (_x, m_start_y), (_x, m_end_y), _line_color, thickness=_thickness)
            cv2.line(to_return_img, (_x, m_end_y), (_x, m_start_y), _line_color, thickness=_thickness)
        text_start_x = _x + 10
        text_start_y = m_start_y + (m_end_y - m_start_y) // 2
        to_return_img = __annotation_text_on_image(to_return_img, (text_start_x, text_start_y), _text_color, m_text)
    for m_y in _y_list:
        cv2.line(to_return_img, (_x - 12, m_y), (_x + 12, m_y), _line_color, thickness=_thickness)
    return to_return_img


def annotation_vertical_height(_img, _x, _start_y, _end_y, _line_color, _text_color, _text):
    return annotation_multi_vertical_height(_img, _x, [_start_y, _end_y], _line_color, _text_color, [_text, ])


def draw_rotated_bbox(_to_draw_image: np.ndarray, _rotated_box: dict, _color: tuple, _thickness: int):
    """
    在图中标注旋转矩形

    Args:
        _to_draw_image: 待标注图像
        _rotated_box:   待标注的旋转矩形（包含center_x,center_y,box_width,box_height,degree）
        _color:     标注颜色（同opencv配置）
        _thickness:     边框粗细（同opencv配置）

    """
    rotated_points = get_coordinates_of_rotated_box(_to_draw_image, _rotated_box)
    cv2.polylines(_to_draw_image, [rotated_points, ], True, _color, _thickness)


def annotate_segmentation(
        _to_draw_image,
        _segmentation_result,
        _background_index=0,
):
    """
    标注分割区域

    Args:
        _to_draw_image:     需要标注的图像
        _segmentation_result:   分割的结果
        _background_index:  背景部分的下标

    Returns:    标注完成的图

    """
    h, w = _to_draw_image.shape[:2]
    if _to_draw_image.shape[:2] != _segmentation_result.shape[:2]:
        _segmentation_result = cv2.resize(_segmentation_result, (w, h), cv2.INTER_NEAREST)
    distinct_index = np.sort(np.unique(_segmentation_result), axis=None)
    candidate_colors = generate_colors(len(distinct_index))
    mask_result_image = _to_draw_image.copy()
    for m_index, m_candidate_color in zip(distinct_index.tolist(), candidate_colors):
        if m_index == _background_index:
            continue
        m_index_segment_result = _segmentation_result == m_index
        np.putmask(mask_result_image, np.repeat(m_index_segment_result[..., None], 3, axis=-1), m_candidate_color)
    add_weighted_result_image = cv2.addWeighted(_to_draw_image, 0.5, mask_result_image, 0.5, 0)
    return add_weighted_result_image


def annotate_detect_rotated_bbox_and_text_result(
        _to_draw_image,
        _rotated_box_list, _text_list,
        _box_color, _box_thickness
):
    """
    标注rotated box和文本到图片上

    Args:
        _to_draw_image:     待标注图像
        _rotated_box_list:  box列表
        _text_list:     文本列表
        _box_color:     box的颜色
        _box_thickness:     box的边框粗细

    Returns:    标注好的图像

    """
    to_return_image = _to_draw_image.copy()
    h, w = to_return_image.shape[:2]
    for m_box, m_text in zip(_rotated_box_list, _text_list):
        draw_rotated_bbox(to_return_image, m_box, _box_color, _box_thickness)
        m_box_center_x = int(m_box['center_x'] * w)
        m_box_center_y = int(m_box['center_y'] * h)
        to_return_image = __annotation_text_on_image(to_return_image,
                                                     (m_box_center_x, m_box_center_y),
                                                     (0, 255, 0),
                                                     m_text['text'])
    return to_return_image
