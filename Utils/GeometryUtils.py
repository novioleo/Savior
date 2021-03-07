import bezier
import numpy as np
import math
import cv2

from scipy.interpolate import splprep, splev


def compute_two_points_angle(_base_point, _another_point):
    """
    以基点作x+延长线，这根线以基点为圆心进行顺时针运动，与基点和另一个点的连线重合所经历的角度
    :param _base_point: 基点
    :param _another_point:  另一个点
    """
    diff_x, diff_y = _another_point[0] - _base_point[0], _another_point[1] - _base_point[1]
    clockwise_angle = 180 + math.degrees(math.atan2(-diff_y, -diff_x))
    return clockwise_angle


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


def nms(_rects, _scores, prob_threshold):
    x1 = _rects[:, 0]
    y1 = _rects[:, 1]
    x2 = _rects[:, 2]
    y2 = _rects[:, 3]
    scores = _scores

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 获得由大到小的分数索引
    score_index = np.argsort(scores)[::-1]

    keep = []

    while len(score_index) > 0:
        max_index = score_index[0]
        # 最大的肯定是需要的框
        keep.append(max_index)
        xx1 = np.maximum(x1[max_index], x1[score_index[1:]])
        yy1 = np.maximum(y1[max_index], y1[score_index[1:]])
        xx2 = np.minimum(x2[max_index], x2[score_index[1:]])
        yy2 = np.minimum(y2[max_index], y2[score_index[1:]])

        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)

        intersection = width * height
        min_areas = areas[score_index[1:]].copy()
        min_areas_mask = areas[score_index[1:]] < areas[max_index]
        min_areas[~min_areas_mask] = areas[max_index]
        iou = intersection / min_areas
        ids = np.where(np.logical_and(iou < prob_threshold, min_areas != intersection))[0]
        # 算iou的时候没把第一个参考框索引考虑进来，所以这里都要+1
        score_index = score_index[ids + 1]
    return keep


def center_rotate_degree_img(_img, _degree, _mask=None):
    if _mask is not None:
        assert _img.shape == _mask.shape, 'mask and shape is not same'
    h, w = _img.shape
    center = (w / 2, h / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center, _degree, 1)
    top_right = np.array((w - 1, 0)) - np.array(center)
    bottom_right = np.array((w - 1, h - 1)) - np.array(center)
    top_right_after_rot = rotate_matrix[0:2, 0:2].dot(top_right)
    bottom_right_after_rot = rotate_matrix[0:2, 0:2].dot(bottom_right)
    new_width = max(int(abs(bottom_right_after_rot[0] * 2) + 0.5), int(abs(top_right_after_rot[0] * 2) + 0.5))
    new_height = max(int(abs(top_right_after_rot[1] * 2) + 0.5), int(abs(bottom_right_after_rot[1] * 2) + 0.5))
    offset_x = (new_width - w) / 2
    offset_y = (new_height - h) / 2
    rotate_matrix[0, 2] += offset_x
    rotate_matrix[1, 2] += offset_y
    rotated_img = cv2.warpAffine(_img, rotate_matrix, (new_width, new_height))
    if _mask is not None:
        rotated_mask = cv2.warpAffine(_mask, rotate_matrix, (new_width, new_height))
    else:
        rotated_mask = None
    return rotated_img, rotated_mask


def resize_convex_hull_polygon(_convex_hull_points, _resize_ratio):
    """
    only for convex hull polygon
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
