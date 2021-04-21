import math
import operator
from functools import reduce

import bezier
import cv2
import numpy as np
import pyclipper
from pyclipper import PyclipperOffset
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon


def compute_two_points_angle(_base_point, _another_point):
    """
    以基点作x轴延长线，这根线以基点为圆心进行顺时针运动，与基点和另一个点的连线重合所经历的角度

    :param _base_point: 基点
    :param _another_point:  另一个点
    """
    diff_x, diff_y = _another_point[0] - _base_point[0], _another_point[1] - _base_point[1]
    clockwise_angle = 180 + math.degrees(math.atan2(-diff_y, -diff_x))
    return clockwise_angle % 360


def get_clockwise_angle_of_two_lines(_center_point, _point_1, _point_2):
    """
    以中心点为圆心，点1到点2之间的顺时针的角度

    :param _center_point:   中心点
    :param _point_1:    点1的坐标
    :param _point_2:    点2的坐标
    :return:    夹角的角度
    """
    angle_1 = compute_two_points_angle(_center_point, _point_1)
    angle_2 = compute_two_points_angle(_center_point, _point_2)
    if angle_2 < angle_1:
        return angle_2 + 360 - angle_1
    else:
        return angle_2 - angle_1


def curved_polygon(_points):
    """
    利用B样条插值对多边形进行优化，使得更加平滑

    :param _points:     多边形所在点
    :return:    平滑后的50个点
    """
    tck, u = splprep(_points.T, u=None, s=1.0, per=1, quiet=2)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array(list(zip(x_new.astype(np.int), y_new.astype(np.int)))).reshape((-1, 1, 2))


def approximate_curved_polygon(_contour, point_num=200):
    """
    使用贝塞尔曲线进行拟合,得到平滑的闭合多边形轮廓

    :param _contour: 构成多边形轮廓的点集. Array:(N, 2)
    :param point_num: 每次拟合的点的数量,越大则越平滑. Int
    :return: 返回平滑后的轮廓点
    """
    to_return_contour = []
    _contour = np.reshape(_contour, (-1, 2))
    # 复制起始点到最后,保证生成闭合的曲线
    _contour = np.vstack((_contour, _contour[0, :].reshape((-1, 2))))
    for start_index in range(0, _contour.shape[0], point_num):
        # 多取一个点,防止曲线中间出现断点
        end_index = start_index + point_num + 1
        end_index = end_index if end_index < _contour.shape[0] else _contour.shape[0]
        nodes = np.transpose(_contour[start_index:end_index, :])
        # 拟合贝塞尔曲线
        curve = bezier.Curve(nodes, degree=nodes.shape[1] - 1)
        curve = curve.evaluate_multi(np.linspace(0.0, 1.0, point_num * 5))
        to_return_contour.append(np.transpose(curve))
    to_return_contour = np.array(to_return_contour).reshape((-1, 2))
    return to_return_contour


def get_region_proportion(_regions, _proportion):
    """
    获取一堆区域的相应的占比
    """
    assert _proportion in {'area', 'height', 'width'}, '不支持的占比计算方式'
    all_region_values = []
    if _proportion == 'area':
        all_region_values = [np.sum(m_region) for m_region in _regions]
    elif _proportion == 'height':
        for m_region in _regions:
            m_region_y, _ = np.where(m_region)
            all_region_values.append(max(m_region_y) - min(m_region_y))
    elif _proportion == 'width':
        for m_region in _regions:
            _, m_region_x = np.where(m_region)
            all_region_values.append(max(m_region_x) - min(m_region_x))
    sum_region_value = sum(all_region_values)
    return [m_region_value / sum_region_value for m_region_value in all_region_values]


def get_bounding_rectangle(_x, _y):
    """
    获得一系列点的组成最小外接矩形的相关信息

    :rtype: object
    :param _x:  一系列点的x值
    :param _y:  一系列点的y值
    :return:    最小外接矩形的左上角x，左上角y，右下角x，右下角y，矩形的高度和宽度
    """
    left_top_corner_x, left_top_corner_y = min(_x), min(_y)
    right_bottom_corner_x, right_bottom_corner_y = max(_x), max(_y)
    width = right_bottom_corner_x - left_top_corner_x
    height = right_bottom_corner_y - left_top_corner_y
    return left_top_corner_x, left_top_corner_y, right_bottom_corner_x, right_bottom_corner_y, height, width


def interpolate_points(_points):
    """
    对线段进行插值，方便后面对多边形进行插值算法的时候更加理想
    :param _points:     所有点
    :return:    插值完成后的点
    """
    to_return_points = []
    _points = np.array(_points)
    for m_point_previous, m_point_next in zip(_points, _points[1:]):
        m_segments = np.max(np.abs(m_point_previous - m_point_next) // 10)
        if m_segments > 1:
            new_x = np.linspace(m_point_previous[0], m_point_next[0], num=int(m_segments), endpoint=False,
                                dtype=np.int)
            new_y = np.linspace(m_point_previous[1], m_point_next[1], num=int(m_segments), endpoint=False,
                                dtype=np.int)
            to_return_points.append(np.vstack([new_x, new_y]))
        else:
            to_return_points.append(np.array([[m_point_previous[0]], [m_point_previous[1]]]))
    return np.hstack(to_return_points).T


def get_polygon_region_contour(_region_mask, _mode='max'):
    """
    获得多边形区域的轮廓

    :param _region_mask:    有多边形区域的图像
    :param _mode:    为'all'时，返回所有的轮廓点集；
                    为'max'时，返回最大的轮廓点集；

    :return:    这个多边形轮廓
    """
    _, contours, _ = cv2.findContours(_region_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    to_return_contours = []
    if _mode == 'max':
        to_return_contours = [max(contours, key=cv2.contourArea), ]
    elif _mode == 'all':
        to_return_contours = contours
    return to_return_contours


def concentric_circle_delete_duplicated(_all_centers, _down_scale_ratio=4):
    """
    简易的二维坐标去重
    相当于将相邻坐标放到一个格子里
    """
    tile_grids = dict()
    to_return_optimized_centers = []
    for m_x, m_y in _all_centers:
        m_x_downscaled, m_y_downscaled = m_x // _down_scale_ratio, m_y // _down_scale_ratio
        m_downscale_name = '%d_%d' % (m_x_downscaled, m_y_downscaled)
        if m_downscale_name not in tile_grids:
            tile_grids[m_downscale_name] = (m_x, m_y, 1)
        else:
            sum_x, sum_y, sum_counter = tile_grids[m_downscale_name]
            tile_grids[m_downscale_name] = (sum_x + m_x, sum_y + m_y, sum_counter + 1)
    for _, (m_sum_x, m_sum_y, m_sum_counter) in tile_grids.items():
        to_return_optimized_centers.append((m_sum_x // m_sum_counter, m_sum_y // m_sum_counter))
    return to_return_optimized_centers


def nms(_rectangles, _scores, _nms_threshold):
    """
    非极大值抑制
    
    Args:
        _rectangles:    所有bbox（非归一化的box）
        _scores:    所有bbox的score
        _nms_threshold: nms的阈值

    Returns:    nms之后的bbox

    """
    x1 = _rectangles[:, 0]
    y1 = _rectangles[:, 1]
    x2 = _rectangles[:, 2]
    y2 = _rectangles[:, 3]
    scores = _scores

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 获得由大到小的分数索引
    score_index = np.argsort(scores)[::-1]

    keep = []

    while len(score_index) > 0:
        max_index = score_index[0]
        # 最大的肯定是需要的框
        keep.append(max_index)
        intersection_left_x = np.maximum(x1[max_index], x1[score_index[1:]])
        intersection_top_y = np.maximum(y1[max_index], y1[score_index[1:]])
        intersection_right_x = np.minimum(x2[max_index], x2[score_index[1:]])
        intersection_bottom_y = np.minimum(y2[max_index], y2[score_index[1:]])

        width = np.maximum(0.0, intersection_right_x - intersection_left_x + 1)
        height = np.maximum(0.0, intersection_bottom_y - intersection_top_y + 1)

        intersection = width * height
        min_areas = areas[score_index[1:]].copy()
        min_areas_mask = areas[score_index[1:]] < areas[max_index]
        min_areas[~min_areas_mask] = areas[max_index]
        iou = intersection / min_areas
        ids = np.where(np.logical_and(iou < _nms_threshold, min_areas != intersection))[0]
        # 算iou的时候没把第一个参考框索引考虑进来，所以这里都要+1
        score_index = score_index[ids + 1]
    return keep


def rotate_points(_points, _degree=0, _center=(0, 0)):
    """
    逆时针绕着一个点旋转点

    Notes:

    points是非归一化的值

    Args:
        _points:    需要旋转的点
        _degree:    角度
        _center:    中心点

    Returns:    旋转后的点

    """
    angle = np.deg2rad(_degree)
    rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
    center = np.atleast_2d(_center)
    points = np.atleast_2d(_points)
    return np.reshape((rotate_matrix @ (points.T - center.T) + center.T).T, (-1, 2))


def get_expand_rotated_points(_image, _center, _rotate_degree):
    """
    图像使用扩张模式旋转之后中心点的位置就会发生变化

    Args:
        _image: 图像
        _center:    旋转中心点
        _rotate_degree: 旋转角度

    Returns:    旋转后的原始的四个点(↖↗↘↙的顺序)，旋转后的中心点

    """
    h, w = _image.shape[:2]
    points = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
        _center
    ])
    rotated_points = rotate_points(points, _rotate_degree, _center)
    offset_x = np.min(rotated_points[:, 0])
    offset_x = -offset_x if offset_x < 0 else 0
    offset_y = np.min(rotated_points[:, 1])
    offset_y = -offset_y if offset_y < 0 else 0
    new_points = rotated_points + [offset_x, offset_y]
    return new_points[:4], new_points[4]


def rotate_degree_img(_img, _degree, _center=None, _with_expand=True, _mask=None):
    """
    逆时针旋转图像

    Args:
        _img:   待旋转图像
        _degree:    角度
        _center:    旋转中心，默认为图像几何中心
        _with_expand:   是否需要调整图像大小，保证所有内容都不丢失
        _mask:      待旋转的mask，可为None

    Returns:    旋转后的图像，旋转后的mask

    """
    if _mask is not None:
        assert _img.shape == _mask.shape[:2], 'mask and shape is not same'
    h, w = _img.shape[:2]
    if _center is None:
        center = (w / 2, h / 2)
    else:
        center = _center
    if _with_expand:
        four_corner_points, _ = get_expand_rotated_points(_img, center, _degree)
        new_width = int(np.max(four_corner_points[:, 0]))
        new_height = int(np.max(four_corner_points[:, 1]))
        current_location = np.array([
            [0, 0],
            [w, 0],
            [w, h],
        ], dtype=np.float32)
        rotate_matrix = cv2.getAffineTransform(current_location, four_corner_points[:3].astype(np.float32))
    else:
        rotate_matrix = cv2.getRotationMatrix2D(center, _degree, 1)
        new_width = w
        new_height = h
    rotated_img = cv2.warpAffine(_img, rotate_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)
    if _mask is not None:
        rotated_mask = cv2.warpAffine(_mask, rotate_matrix, (new_width, new_height), flags=cv2.INTER_NEAREST)
    else:
        rotated_mask = None
    return rotated_img, rotated_mask


def resize_convex_hull_polygon(_convex_hull_points, _resize_ratio):
    """
    对凸包的多边形进行缩放

    Args:
        _convex_hull_points:    凸包多边形的轮廓
        _resize_ratio:  缩放比例

    Returns:    缩放后的点

    """
    center_point = np.mean(_convex_hull_points, axis=0)
    diff_points = _convex_hull_points - center_point
    r = np.linalg.norm(diff_points, axis=1)
    theta = np.arctan2(diff_points[:, 1], diff_points[:, 0])
    target_r = r * _resize_ratio
    to_return_points = np.zeros_like(diff_points, dtype=np.float)
    to_return_points[:, 0] = target_r * np.cos(theta)
    to_return_points[:, 1] = target_r * np.sin(theta)
    # 向下取整
    return (to_return_points + center_point).astype(np.int)


def get_distance(_p1, _p2):
    """
    获取两点之间的欧式距离

    Args:
       _p1:  点的坐标
       _p2:  点的坐标

    Returns:   两点间的欧式距离

    """
    return np.sqrt(np.sum(np.square(_p1 - _p2)))


def resize_with_height(_image, _target_height):
    """
    将图像高度resize到指定高度的等比例缩放

    Args:
        _image:     待缩放图像
        _target_height:     目标高度

    Returns:    缩放后的图像

    """
    h, w = _image.shape[:2]
    ratio = h / _target_height
    target_w = int(np.ceil(w / ratio))
    return cv2.resize(_image, (target_w, _target_height))


def resize_with_width(_image, _target_width):
    """
    将图像宽度resize到指定宽度的等比例缩放

    Args:
        _image:     待缩放图像
        _target_width:     目标宽度

    Returns:    缩放后的图像

    """
    h, w = _image.shape[:2]
    ratio = w / _target_width
    target_h = int(np.ceil(h / ratio))
    return cv2.resize(_image, (_target_width, target_h))


def resize_with_short_side(_image, _target_short_side_size):
    """
    将图像最短边resize到指定长度的等比例缩放

    Args:
        _image:     图像
        _target_short_side_size:    最短边目标长度

    Returns:    缩放后的图像

    """
    h, w = _image.shape[:2]
    if h > w:
        return resize_with_width(_image, _target_short_side_size)
    else:
        return resize_with_height(_image, _target_short_side_size)


def resize_with_long_side(_image, _target_long_side_size):
    """
    将图像最长边resize到指定长度的等比例缩放

    Args:
        _image:     图像
        _target_long_side_size:    最长边目标长度

    Returns:    缩放后的图像

    """
    h, w = _image.shape[:2]
    if h > w:
        return resize_with_height(_image, _target_long_side_size)
    else:
        return resize_with_width(_image, _target_long_side_size)


def _compute_image_specific_base(_image, _height_base=None, _width_base=None):
    """
    计算图像的宽高在一定基数基础上的最邻近向上取整的宽高

    Args:
        _image:     图像
        _height_base:   高度的基数
        _width_base:    宽度的基数

    Returns:    最临近高度，最邻近宽度

    """
    h, w = _image.shape[:2]
    target_h = h
    target_w = w
    if _height_base is not None:
        if h <= _height_base:
            target_h = _height_base
        else:
            target_h = int(np.ceil(h / _height_base) * _height_base)
    if _width_base is not None:
        if w <= _width_base:
            target_w = _width_base
        else:
            target_w = int(np.ceil(w / _width_base) * _width_base)
    return target_h, target_w


def resize_with_specific_base(_image, _height_base=None, _width_base=None):
    """
    将图像缩放到特定基的倍数的高宽

    Args:
        _image:     待缩放图像
        _height_base:   高度的基
        _width_base:    宽度的基

    Returns:    缩放后的图像

    """
    target_h, target_w = _compute_image_specific_base(_image, _height_base, _width_base)
    return cv2.resize(_image, (target_w, target_h))


def center_pad_image_with_specific_base(_image, _height_base=None, _width_base=None, _pad_value=0,
                                        _output_pad_ratio=False):
    """
    将图像中心填充到特定基的倍数的高宽的图像中

    Args:
        _image:     待缩放图像
        _height_base:   高度的基
        _width_base:    宽度的基
        _pad_value:     pad的填充值
        _output_pad_ratio:  是否输出pad（width_pad,height_pad）的占比，方便后面在计算的时候减去对应的值

    Returns:    缩放后的图像

    """
    h, w = _image.shape[:2]
    target_h, target_w = _compute_image_specific_base(_image, _height_base, _width_base)
    if len(_image.shape) == 3:
        full_size_image = np.ones((target_h, target_w, _image.shape[2]), dtype=_image.dtype) * _pad_value
    else:
        full_size_image = np.ones((target_h, target_w), dtype=_image.dtype) * _pad_value
    left_margin = (target_w - w) // 2
    right_margin = left_margin + w
    top_margin = (target_h - h) // 2
    bottom_margin = top_margin + h
    full_size_image[top_margin:bottom_margin, left_margin:right_margin, ...] = _image
    if not _output_pad_ratio:
        return full_size_image
    else:
        return full_size_image, (left_margin / target_w, top_margin / target_h)


def remove_image_pad(_padded_image, _original_image, _left_margin, _top_margin):
    """
    移除图像的pad

    Args:
        _padded_image:  已经pad后的图像
        _original_image:    原图
        _left_margin:   左边界
        _top_margin:    上边界

    Returns:    移除边界后的图

    """
    padded_h, padded_w = _padded_image.shape[:2]
    original_h, original_w = _original_image.shape[:2]
    left_margin_pixels = int(_left_margin * padded_w)
    top_margin_pixels = int(_top_margin * padded_h)
    right_boundary = left_margin_pixels + original_w
    bottom_boundary = top_margin_pixels + original_h
    return _padded_image[top_margin_pixels:bottom_boundary, left_margin_pixels:right_boundary, ...]


def get_cropped_image(_image, _location):
    """
    抠取图中的特定区域

    Args:
        _image:     待抠取图像
        _location:  待抠取区域

    Returns:    抠取出来的结果

    """
    h, w = _image.shape[:2]
    top_left_x = int(np.clip(_location['top_left_x'], a_min=0, a_max=1) * w)
    top_left_y = int(np.clip(_location['top_left_y'], a_min=0, a_max=1) * h)
    bottom_right_x = int(np.clip(_location['bottom_right_x'], a_min=0, a_max=1) * w)
    bottom_right_y = int(np.clip(_location['bottom_right_y'], a_min=0, a_max=1) * h)
    return _image.copy()[top_left_y:bottom_right_y + 1, top_left_x:bottom_right_x + 1, ...]


def get_min_area_bbox(_image, _contour, _scale_ratio=1.0):
    """
    获取一个contour对应的最小面积矩形
    note:主要是解决了旋转角度不合适的问题

    Args:
        _image:     bbox所在图像
        _contour:   轮廓
        _scale_ratio:      缩放比例


    Returns:    最小面积矩形的相关信息

    """
    h, w = _image.shape[:2]
    if _scale_ratio != 1:
        reshaped_contour = _contour.reshape(-1, 2)
        current_polygon = Polygon(reshaped_contour)
        distance = current_polygon.area * _scale_ratio / current_polygon.length
        offset = PyclipperOffset()
        offset.AddPath(reshaped_contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        all_paths = offset.Execute(distance)
        if len(all_paths) > 0:
            max_path = max(all_paths, key=lambda x: cv2.contourArea(np.array(x)))
            scaled_contour = np.array(max_path).reshape(-1, 1, 2)
        else:
            return None
    else:
        scaled_contour = _contour
    try:
        # 会存在contour不合法的情况下，无法计算得到最小面积矩形
        rotated_box = cv2.minAreaRect(scaled_contour)
        if -90 <= rotated_box[2] <= -45:
            to_rotate_degree = rotated_box[2] + 90
            bbox_height, bbox_width = rotated_box[1]
        else:
            to_rotate_degree = rotated_box[2]
            bbox_width, bbox_height = rotated_box[1]
        # 几何信息归一化可以方便进行在缩放前的图像上进行操作
        to_return_rotated_box = {
            'degree': int(to_rotate_degree),
            'center_x': rotated_box[0][0] / w,
            'center_y': rotated_box[0][1] / h,
            'box_height': bbox_height / h,
            'box_width': bbox_width / w,
        }
        return to_return_rotated_box
    except Exception as e:
        return None


def get_rotated_box_roi_from_image(_image, _rotated_box, _scale_ratio=1.0):
    """
    在图像中抠取一个旋转的box的roi

    Args:
        _image:     待抠取图像
        _rotated_box:   旋转的box
        _scale_ratio:   缩放比例

    Returns:    抠取的roi

    """
    h, w = _image.shape[:2]
    to_rotate_degree = _rotated_box['degree']
    box_center = (_rotated_box['center_x'] * w, _rotated_box['center_y'] * h)
    half_box_height, half_box_width = \
        _rotated_box['box_height'] * _scale_ratio / 2, _rotated_box['box_width'] * _scale_ratio / 2
    if to_rotate_degree != 0:
        rotated_image, _ = rotate_degree_img(_image, to_rotate_degree, box_center, _with_expand=False)
    else:
        rotated_image = _image.copy()
    to_crop_location = {
        'top_left_x': _rotated_box['center_x'] - half_box_width,
        'top_left_y': _rotated_box['center_y'] - half_box_height,
        'bottom_right_x': _rotated_box['center_x'] + half_box_width,
        'bottom_right_y': _rotated_box['center_y'] + half_box_height,
    }
    cropped_image = get_cropped_image(rotated_image, to_crop_location)
    return cropped_image


def get_coordinates_of_rotated_box(_image, _rotated_box):
    """
    获取旋转的矩形的对应的四个顶点坐标

    Args:
        _image:     对应的图像
        _rotated_box:   旋转的矩形

    Returns:    四个对应在图像中的坐标点

    """
    h, w = _image.shape[:2]
    center_x = _rotated_box['center_x']
    center_y = _rotated_box['center_y']
    half_box_width = _rotated_box['box_width'] / 2
    half_box_height = _rotated_box['box_height'] / 2
    raw_points = np.array([
        [center_x - half_box_width, center_y - half_box_height],
        [center_x + half_box_width, center_y - half_box_height],
        [center_x + half_box_width, center_y + half_box_height],
        [center_x - half_box_width, center_y + half_box_height]
    ]) * (w, h)
    rotated_points = rotate_points(raw_points, _rotated_box['degree'], (center_x * w, center_y * h))
    rotated_points[:, 0] = np.clip(rotated_points[:, 0], a_min=0, a_max=w)
    rotated_points[:, 1] = np.clip(rotated_points[:, 1], a_min=0, a_max=h)
    return rotated_points.astype(np.int32)


def clockwise_sort_points(_point_coordinates):
    """
        以左上角为起点的顺时针排序
        原理就是将笛卡尔坐标转换为极坐标，然后对极坐标的φ进行排序
    Args:
        _point_coordinates:  待排序的点[(x,y),]

    Returns:    排序完成的点

    """
    center_point = tuple(
        map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), _point_coordinates),
            [len(_point_coordinates)] * 2))
    return sorted(_point_coordinates, key=lambda coord: (180 + math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center_point))[::-1]))) % 360)


def force_convert_image_to_bgr(_image):
    """
    将图像转换为bgr

    Args:
        _image:     待转换图像

    Returns:    转换后的图像

    """

    if len(_image.shape) == 2:
        candidate_image = cv2.cvtColor(_image, cv2.COLOR_GRAY2BGR)
    else:
        if _image.shape[-1] == 4:
            candidate_image = cv2.cvtColor(_image, cv2.COLOR_BGRA2BGR)
        else:
            candidate_image = _image
    return candidate_image


def face_align(_image, _landmark, _target_shape):
    """
    人脸对齐

    Args:
        _image: 人脸图片
        _landmark:  人脸图片上的landmark
        _target_shape:  目标尺寸

    Returns:    对齐后的人脸

    """
    reference_facial_points = np.array([
        [0.31556875, 0.4615741],
        [0.6826229, 0.45983392],
        [0.5002625, 0.6405054],
        [0.3494719, 0.82469195],
        [0.6534365, 0.8232509]
    ], dtype=np.float32)
    target_facial_points = reference_facial_points.copy() * _target_shape
    h, w = _image.shape[:2]
    remapped_landmark = _landmark.copy() * [w, h]
    transform_matrix = cv2.estimateRigidTransform(remapped_landmark, target_facial_points, True)
    face_img = cv2.warpAffine(_image, transform_matrix, _target_shape)
    return face_img


def correct_face_orientation(_image, _landmark_info):
    """
    校正人脸的方向

    Args:
        _image: 人脸照片
        _landmark_info:     landmark信息

    Returns:    旋转后的人脸照片，以及反变化的回调函数

    """
    h, w = _image.shape[:2]
    reference_facial_points = np.array([
        [30.29459953, 51.69630003],
        [65.53179932, 51.50139904],
        [48.02519989, 71.73660183],
    ], dtype=np.float32)
    if _landmark_info['points_count'] == 5:
        points_index = [0, 1, 2]
    elif _landmark_info['points_count'] == 106:
        points_index = [38, 88, 86]
    else:
        raise NotImplementedError(f"Cannot correct face with {_landmark_info['points_count']} landmark points now")
    landmark_x = _landmark_info['x_locations'][points_index]
    landmark_y = _landmark_info['y_locations'][points_index]

    landmark = np.stack([landmark_x, landmark_y], axis=1)
    transform_matrix = cv2.getAffineTransform((landmark * [96.0, 112.0]).astype(np.float32), reference_facial_points)
    center_point = (landmark[-1] * (w, h)).astype(np.float32)
    degree = math.degrees(math.atan(transform_matrix[0, 0] / transform_matrix[0, 1]))
    assert -90 <= degree <= 90, 'the face correct angle must be between -90 degree and 90 degree'
    if degree > 0:
        rotation_degree = degree - 90
    else:
        rotation_degree = degree + 90
    rotated_image, _ = rotate_degree_img(_image, rotation_degree, (center_point[0], center_point[1]), True)
    rotated_points, rotated_center_point = get_expand_rotated_points(_image, (center_point[0], center_point[1]),
                                                                     rotation_degree)
    original_h, original_w = _image.shape[:2]
    transform_back_matrix = cv2.getAffineTransform(
        rotated_points[:3].astype(np.float32),
        np.array([[0, 0], [original_w - 1, 0], [original_w - 1, original_h - 1]], dtype=np.float32)
    )

    def _rotate_back_callback(_to_rotate_back_image):
        """
        用于将需要进行反变化回原样的图回调函数
        
        Args:
            _to_rotate_back_image:  待反变化的图

        Returns:    与原图对应

        """
        return cv2.warpAffine(_to_rotate_back_image,
                              transform_back_matrix,
                              (original_w, original_h),
                              flags=cv2.INTER_NEAREST)

    return rotated_image, _rotate_back_callback
