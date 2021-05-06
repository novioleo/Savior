import cv2
import numpy as np

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Operators.ExampleQRCodeDetectOperator.PostProcess import generate_prior_boxes, ssd_detect
from Utils.GeometryUtils import center_pad_image_with_specific_base, force_convert_image_to_gray, resize_with_long_side
from Utils.InferenceHelpers import TritonInferenceHelper


class QRCodeDetectWithSSD(DummyAlgorithmWithModel):
    name = '微信的基于mbv2的ssd的二维码检测'
    __version__ = 'v1.0.20210505'

    def __init__(self, _inference_config, _is_test,
                 _candidate_size=384, _score_threshold=0.2, _iou_threshold=0.45,
                 ):
        """
        需要根据图像的实际情况进行参数调整

        Args:
            _inference_config:  推理用helper的配置
            _is_test:   是否是测试模式
            _candidate_size:    推理使用的图像大小
            _score_threshold:
            _iou_threshold:
        """
        super().__init__(_inference_config, _is_test)
        self.score_threshold = _score_threshold
        self.iou_threshold = _iou_threshold
        assert _candidate_size % 32 == 0 and _candidate_size >= 384, 'candidate size必须为大于384的32的倍数'
        self.candidate_size = _candidate_size
        self.variance = [0.1, 0.2]
        self.num_classes = 2

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            inference_helper = TritonInferenceHelper('QRCodeDetect',
                                                     self.inference_config['triton_url'],
                                                     self.inference_config['triton_port'],
                                                     'QRCodeDetect', 1)
            inference_helper.add_image_input('INPUT__0', (-1, -1, 1), '检测用灰度的图像',
                                             ([0, ], [255, ]))
            inference_helper.add_output('OUTPUT__0', (-1), 'anchor location')
            inference_helper.add_output('OUTPUT__1', (-1), 'anchor score')
            self.inference_helper = inference_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for qrcode detect not implement")

    def execute(self, _image):
        to_return_result = {
            'locations': [],
        }
        resized_image = resize_with_long_side(_image, self.candidate_size)
        resized_h, resized_w = resized_image.shape[:2]
        # 保证输入网络中的图像为矩形
        padded_image, (width_pad_ratio, height_pad_ratio) = center_pad_image_with_specific_base(
            resized_image,
            _width_base=self.candidate_size,
            _height_base=self.candidate_size,
            _output_pad_ratio=True
        )
        candidate_image = force_convert_image_to_gray(padded_image)
        candidate_image_h, candidate_image_w = candidate_image.shape[:2]
        if isinstance(self.inference_helper, TritonInferenceHelper):
            result = self.inference_helper.infer(_need_tensor_check=False, INPUT__0=candidate_image.astype(np.float32))
            box_location = result['OUTPUT__0'].squeeze(0)
            box_confidence = result['OUTPUT__1'].squeeze(0)
        else:
            raise NotImplementedError(
                f"{self.inference_helper.type_name} helper for qrcode detect not implement")

        stage4_prior_boxes = generate_prior_boxes(
            candidate_image_h // 16, candidate_image_w // 16,
            candidate_image_h, candidate_image_w,
            _min_size=50, _max_size=100,
            _aspect_ratios=[2.0, 0.5, 3.0, 0.3],
            _flip=False, _clip=False,
            _variance=[0.1, 0.1, 0.2, 0.2],
            _step=16,
            _offset=0.5,
        )
        stage5_prior_boxes = generate_prior_boxes(
            candidate_image_h // 32, candidate_image_w // 32,
            candidate_image_h, candidate_image_w,
            _min_size=100, _max_size=150,
            _aspect_ratios=[2.0, 0.5, 3.0, 0.3],
            _flip=False, _clip=False,
            _variance=[0.1, 0.1, 0.2, 0.2],
            _step=32,
            _offset=0.5,
        )
        stage6_prior_boxes = generate_prior_boxes(
            candidate_image_h // 32, candidate_image_w // 32,
            candidate_image_h, candidate_image_w,
            _min_size=150, _max_size=200,
            _aspect_ratios=[2.0, 0.5, 3.0, 0.3],
            _flip=False, _clip=False,
            _variance=[0.1, 0.1, 0.2, 0.2],
            _step=32,
            _offset=0.5,
        )
        stage7_prior_boxes = generate_prior_boxes(
            candidate_image_h // 32, candidate_image_w // 32,
            candidate_image_h, candidate_image_w,
            _min_size=200, _max_size=300,
            _aspect_ratios=[2.0, 0.5, 3.0, 0.3],
            _flip=False, _clip=False,
            _variance=[0.1, 0.1, 0.2, 0.2],
            _step=32,
            _offset=0.5,
        )
        stage8_prior_boxes = generate_prior_boxes(
            candidate_image_h // 32, candidate_image_w // 32,
            candidate_image_h, candidate_image_w,
            _min_size=300, _max_size=400,
            _aspect_ratios=[2.0, 0.5, 3.0, 0.3],
            _flip=False, _clip=False,
            _variance=[0.1, 0.1, 0.2, 0.2],
            _step=32,
            _offset=0.5,
        )
        all_stage_prior_boxes = np.concatenate([
            stage4_prior_boxes,
            stage5_prior_boxes,
            stage6_prior_boxes,
            stage7_prior_boxes,
            stage8_prior_boxes
        ], axis=1)
        detect_result = ssd_detect(candidate_image_h,
                                   candidate_image_w,
                                   box_location,
                                   box_confidence,
                                   all_stage_prior_boxes,
                                   2, self.variance, self.score_threshold, self.iou_threshold)[0]
        height_resize_ratio = candidate_image_h / resized_h
        width_resize_ratio = candidate_image_w / resized_w
        for m_detect_qrcode in detect_result:
            m_detect_bbox_width = (m_detect_qrcode[2] - m_detect_qrcode[0]) * width_resize_ratio
            m_detect_bbox_height = (m_detect_qrcode[2] - m_detect_qrcode[0]) * height_resize_ratio
            m_detect_bbox_top_left_x = (m_detect_qrcode[0] - width_pad_ratio) * width_resize_ratio
            m_detect_bbox_top_left_y = (m_detect_qrcode[1] - height_pad_ratio) * height_resize_ratio
            to_return_result['locations'].append({
                'box_width': m_detect_bbox_width,
                'box_height': m_detect_bbox_height,
                'center_x': m_detect_bbox_top_left_x + m_detect_bbox_width / 2,
                'center_y': m_detect_bbox_top_left_y + m_detect_bbox_height / 2,
                'degree': 0,
            })
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    from Utils.AnnotationTools import draw_rotated_bbox

    ag = ArgumentParser('QRCode Detect Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
    qrcode_detect_handler = QRCodeDetectWithSSD({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port
    }, True, 384, 0.2, 0.45)
    detected_code_result = qrcode_detect_handler.execute(img)
    to_draw_image = img.copy()
    for m_location in detected_code_result['locations']:
        draw_rotated_bbox(to_draw_image, m_location, (0, 255, 0), 3)
    cv2.imshow('to_draw_image', to_draw_image)
    cv2.waitKey(0)
