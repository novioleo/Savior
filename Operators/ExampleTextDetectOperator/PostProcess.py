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
        to_return_scores.append(np.sum(available_region) / np.sum(mask_region))
    return to_return_boxes, to_return_scores
