import numpy as np
import math

from Utils.GeometryUtils import nms


def generate_prior_boxes(_feature_height, _feature_width,
                         _image_height, _image_width,
                         _min_size, _max_size,
                         _aspect_ratios,
                         _flip, _clip,
                         _step,
                         _offset,
                         _variance,
                         ):
    mw = float(_min_size) / _image_width / 2
    mh = float(_min_size) / _image_height / 2
    ww = math.sqrt(mw * float(_max_size) / _image_width) / 2
    hh = math.sqrt(mh * float(_max_size) / _image_height) / 2
    center_x = np.repeat(np.arange(_feature_width)[None, ...], _feature_height, axis=0) + _offset
    center_y = center_x.copy().T
    center_x = center_x * _step / _image_width
    center_y = center_y * _step / _image_height
    mean = [center_x - mw, center_y - mh, center_x + mw, center_y + mh]
    if _max_size > _min_size:
        mean += [center_x - ww, center_y - hh, center_x + ww, center_y + hh]
        for m_aspect_ratio in _aspect_ratios:
            m_sqrt_aspect_ratio = math.sqrt(m_aspect_ratio)
            ww = mw * m_sqrt_aspect_ratio
            hh = mh / m_sqrt_aspect_ratio
            mean += [center_x - ww, center_y - hh, center_x + ww, center_y + hh]
            if _flip:
                ww = mw / m_sqrt_aspect_ratio
                hh = mh * m_sqrt_aspect_ratio
                mean += [center_x - ww, center_y - hh, center_x + ww, center_y + hh]
    mean_tensor = np.stack(mean, axis=-1)
    output1 = mean_tensor.reshape((-1, 4))
    output2 = np.ones_like(output1) * _variance
    if _clip:
        output1.clip(max=1, min=0)
    output1 = output1.reshape((1, -1))
    output2 = output2.reshape((1, -1))
    output = np.concatenate([output1, output2], axis=0)
    return output


def center_size(_boxes):
    return np.concatenate([(_boxes[:, 2:] + _boxes[:, :2]) / 2, _boxes[:, 2:] - _boxes[:, :2]], 1)


def bbox_decode(_locations, _prior_boxes, _variances):
    boxes = np.concatenate((
        _prior_boxes[:, :2] + _locations[:, :2] * _variances[0] * _prior_boxes[:, 2:],
        _prior_boxes[:, 2:] * np.exp(_locations[:, 2:] * _variances[1]))
        , axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def ssd_detect(_input_tensor_height, _input_tensor_width,
               _locations: np.ndarray, _confidences: np.ndarray,
               _prior_boxes: np.ndarray, _num_classes: int,
               _variance=None, _confidence_threshold=0.2, _nms_threshold=0.45):
    if _variance is None:
        _variance = [0.1, 0.2]
    location_data = _locations.reshape((-1, 4))
    prior_data = center_size(_prior_boxes[0, :].reshape((-1, 4)))
    confidence_data = _confidences.reshape((-1, _num_classes)).T
    decoded_boxes = bbox_decode(location_data, prior_data, _variance)
    to_return_all_classes_detect_result = []
    for i in range(1, _num_classes):
        m_available_confidence_mask = confidence_data[i] > _confidence_threshold
        if np.sum(m_available_confidence_mask) == 0:
            to_return_all_classes_detect_result.append([])
        m_scores = confidence_data[i][m_available_confidence_mask]
        m_boxes = decoded_boxes[m_available_confidence_mask]
        m_boxes_original = m_boxes * [_input_tensor_width,
                                      _input_tensor_height,
                                      _input_tensor_width,
                                      _input_tensor_height]
        m_filtered_box_ids = nms(m_boxes_original, m_scores, _nms_threshold)
        to_return_all_classes_detect_result.append(m_boxes[m_filtered_box_ids])
    return to_return_all_classes_detect_result
