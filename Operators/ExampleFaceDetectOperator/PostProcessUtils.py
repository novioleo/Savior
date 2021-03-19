import numpy as np


def shift(_shape, _stride, _anchors):
    shift_x = (np.arange(0, _shape[1]) + 0.5) * _stride
    shift_y = (np.arange(0, _shape[0]) + 0.5) * _stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K * A, 4) shifted anchors
    A = _anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(_base_size=16, _scales=None):
    if _scales is None:
        _scales = np.array([
            2 ** 0,
            2 ** (1.0 / 3.0),
            2 ** (2.0 / 3.0)
        ])

    num_anchors = len(_scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = _base_size * np.tile(_scales, (2, 1)).T

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def get_anchors(_image_shape):
    pyramid_levels = [3, 4, 5]
    strides = [2 ** x for x in pyramid_levels]
    sizes = [2 ** 4.0, 2 ** 6.0, 2 ** 8.0]
    scales = np.array([2 ** 0, 2 ** (1 / 2.0), 2 ** 1.0])
    image_shapes = [(_image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    all_anchors = np.zeros((0, 4)).astype(np.float32)

    for m_level_index, m_level in enumerate(pyramid_levels):
        anchors = generate_anchors(_base_size=int(sizes[m_level_index]), _scales=scales)
        shifted_anchors = shift(image_shapes[m_level_index], strides[m_level_index], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors.astype(np.float32)


def regress_boxes(_anchors, _bbox_deltas, _landmark_deltas, _image_shape):
    h, w = _image_shape
    std_landmark = np.ones((1, 10)) * 0.1
    std_box = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)

    widths = _anchors[..., 2] - _anchors[..., 0]
    heights = _anchors[..., 3] - _anchors[..., 1]
    center_x = _anchors[..., 0] + 0.5 * widths
    center_y = _anchors[..., 1] + 0.5 * heights

    _bbox_deltas = _bbox_deltas * std_box

    bbox_dx = _bbox_deltas[..., 0]
    bbox_dy = _bbox_deltas[..., 1]
    bbox_dw = _bbox_deltas[..., 2]
    bbox_dh = _bbox_deltas[..., 3]

    # get predicted boxes
    predict_center_x = center_x + bbox_dx * widths
    predict_center_y = center_y + bbox_dy * heights
    half_predict_width = np.exp(bbox_dw) * widths / 2
    half_predict_height = np.exp(bbox_dh) * heights / 2

    predict_boxes_top_left_x = np.clip(predict_center_x - half_predict_width, a_min=0, a_max=w) / w
    predict_boxes_top_left_y = np.clip(predict_center_y - half_predict_height, a_min=0, a_max=h) / h
    predict_boxes_bottom_right_x = np.clip(predict_center_x + half_predict_width, a_min=0, a_max=w) / w
    predict_boxes_bottom_right_y = np.clip(predict_center_y + half_predict_height, a_min=0, a_max=h) / h

    predict_boxes = np.stack([predict_boxes_top_left_x,
                              predict_boxes_top_left_y,
                              predict_boxes_bottom_right_x,
                              predict_boxes_bottom_right_y], axis=-1)
    if _landmark_deltas is not None:
        _landmark_deltas = _landmark_deltas * std_landmark
        landmark_points_x = np.clip(center_x + _landmark_deltas[..., ::2] * widths, a_min=0, a_max=w) / w
        landmark_points_y = np.clip(center_y + _landmark_deltas[..., 1::2] * heights, a_min=0, a_max=h) / h
        predict_landmarks = np.concatenate([landmark_points_x, landmark_points_y], axis=-1)
    else:
        predict_landmarks = None
    return predict_boxes, predict_landmarks
