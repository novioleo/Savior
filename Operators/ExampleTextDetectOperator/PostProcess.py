import cv2
import numpy as np

from Utils.GeometryUtils import get_min_area_bbox


def db_post_process(_predict_score, _thresh, _bbox_scale_ratio, _min_size=5):
    instance_score = _predict_score.squeeze()
    h, w = instance_score.shape[:2]
    available_region = np.zeros_like(instance_score, dtype=np.float32)
    np.putmask(available_region, instance_score > _thresh, instance_score)
    to_return_boxes = []
    to_return_scores = []
    mask_region = (available_region > 0).astype(np.uint8) * 255
    structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    refined_mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_CLOSE, structure_element)
    if cv2.__version__.startswith('3'):
        _, contours, _ = cv2.findContours(refined_mask_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    elif cv2.__version__.startswith('4'):
        contours, _ = cv2.findContours(refined_mask_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        raise NotImplementedError(f'opencv {cv2.__version__} not support')
    for m_contour in contours:
        if len(m_contour) < 4 and cv2.contourArea(m_contour) < 16:
            continue
        m_rotated_box = get_min_area_bbox(refined_mask_region, m_contour, _bbox_scale_ratio)
        if m_rotated_box is None:
            continue
        m_box_width = m_rotated_box['box_width']
        m_box_height = m_rotated_box['box_height']
        if min(m_box_width * w, m_box_height * h) < _min_size:
            continue
        to_return_boxes.append(m_rotated_box)
        m_available_mask = np.zeros_like(available_region, dtype=np.uint8)
        cv2.drawContours(m_available_mask, [m_contour,],0, 255, thickness=-1)
        m_region_mask = cv2.bitwise_and(available_region, available_region, mask=m_available_mask)
        m_mask_count = np.count_nonzero(m_available_mask)
        to_return_scores.append(float(np.sum(m_region_mask) / m_mask_count))
    return to_return_boxes, to_return_scores
