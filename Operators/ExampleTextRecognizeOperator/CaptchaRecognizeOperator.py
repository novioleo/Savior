import os

import numpy as np
from scipy.special import softmax

from Operators.DummyAlgorithmWithModel import DummyAlgorithmWithModel
from Utils.GeometryUtils import force_convert_image_to_bgr, resize_with_height, pad_image_with_specific_base
from Utils.InferenceHelpers import TritonInferenceHelper


class Captcha1RecognizeWithMaster(DummyAlgorithmWithModel):
    """
    基于Master对于验证码种类1进行识别
    训练使用的master的版本：https://github.com/novioleo/MASTER-pytorch 持续更新
    """
    name = "基于Master的常用验证码种类1识别"
    __version__ = 'v1.0.20210515'

    def __init__(self, _inference_config, _alphabet_config_name, _is_test):
        self.encoder_inference_helper = None
        self.decoder_inference_helper = None
        super().__init__(_inference_config, _is_test)
        self.target_height = 48
        self.target_width = 160
        self.probability_threshold = 0.8
        alphabet_file_path = os.path.join(os.path.dirname(__file__), 'assets', _alphabet_config_name + '.txt')
        with open(alphabet_file_path, mode='r') as to_read_alphabet:
            self.keys = [m_line.strip() for m_line in to_read_alphabet]

    def get_inference_helper(self):
        if self.inference_config['name'] == 'triton':
            encoder_helper = TritonInferenceHelper(
                'Captcha1RecognizeEncoder',
                self.inference_config['triton_url'],
                self.inference_config['triton_port'],
                'Captcha1RecognizeEncoder',
                1
            )
            decoder_helper = TritonInferenceHelper(
                'Captcha1RecognizeDecoder',
                self.inference_config['triton_url'],
                self.inference_config['triton_port'],
                'Captcha1RecognizeDecoder',
                1
            )
            encoder_helper.add_image_input('INPUT__0', (48, 160, 3), '识别用的图像',
                                           ([127.5, 127.5, 127.5], [127.5, 127.5, 127.5]))
            encoder_helper.add_output('OUTPUT__0', (240, 512), 'memory')
            decoder_helper.add_input('INPUT__0', (-1,), '已预测的label')
            decoder_helper.add_input('INPUT__1', (240, 512), 'memory')
            decoder_helper.add_output('OUTPUT__0', (-1, -1), '预测的label')
            self.encoder_inference_helper = encoder_helper
            self.decoder_inference_helper = decoder_helper
        else:
            raise NotImplementedError(
                f"{self.inference_config['name']} helper for captcha 1 recognize with master not implement")

    def predict(self, _memory, _max_length, _sos_symbol, _padding_symbol):
        batch_size = 1
        to_return_label = np.ones((batch_size, _max_length + 2), dtype=np.int64) * _padding_symbol
        probabilities = np.ones((batch_size, _max_length + 2), dtype=np.float32)
        to_return_label[:, 0] = _sos_symbol
        for i in range(_max_length + 1):
            if isinstance(self.decoder_inference_helper, TritonInferenceHelper):
                result = self.decoder_inference_helper.infer(_need_tensor_check=False,
                                                             INPUT__0=to_return_label,
                                                             INPUT__1=_memory)
                m_label = result['OUTPUT__0']
            else:
                raise NotImplementedError(
                    f"{self.decoder_inference_helper.type_name} helper for captcha 1 recognize decoder not implement")
            m_probability = softmax(m_label, axis=-1)
            m_next_word = np.argmax(m_probability, axis=-1)
            m_max_probs = np.max(m_probability, axis=-1)
            to_return_label[:, i + 1] = m_next_word[:, i]
            probabilities[:, i + 1] = m_max_probs[:, i]
        return to_return_label.squeeze(0), probabilities.squeeze(0)

    def execute(self, _image):
        to_return_result = {
            'text': '',
            'probability': 1.0
        }
        bgr_image = force_convert_image_to_bgr(_image)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized_image = resize_with_height(rgb_image, self.target_height)
        candidate_image = pad_image_with_specific_base(resized_image, 0, 0,
                                                       _width_base=self.target_width, _pad_value=0)
        if isinstance(self.encoder_inference_helper, TritonInferenceHelper):
            result = self.encoder_inference_helper.infer(_need_tensor_check=False,
                                                         INPUT__0=candidate_image.astype(np.float32))
            memory = result['OUTPUT__0']
        else:
            raise NotImplementedError(
                f"{self.encoder_inference_helper.type_name} helper for captcha 1 recognize encoder not implement")
        candidate_label_length = 10
        # sos:2 eos:1 pad:0 unk:3
        label, label_probability = self.predict(memory, candidate_label_length, 2, 0)
        total_probability = 0
        for m_label, m_probability in zip(label, label_probability):
            # 包括了unk,sos,eos,pad，所以要删除
            if m_probability >= self.probability_threshold and m_label >= 4:
                to_return_result['text'] += self.keys[m_label - 4]
                total_probability += m_probability
        to_return_result['probability'] = total_probability / len(to_return_result['text'])
        return to_return_result


if __name__ == '__main__':
    from argparse import ArgumentParser
    import cv2
    from pprint import pprint

    ag = ArgumentParser('Captcha Recognize Example')
    ag.add_argument('-i', '--image_path', dest='image_path', type=str, required=True, help='本地图像路径')
    ag.add_argument('-u', '--triton_url', dest='triton_url', type=str, required=True, help='triton url')
    ag.add_argument('-p', '--triton_port', dest='triton_port', type=int, default=8001, help='triton grpc 端口')
    args = ag.parse_args()
    img = cv2.imread(args.image_path)
    master_handler = Captcha1RecognizeWithMaster({
        'name': 'triton',
        'triton_url': args.triton_url,
        'triton_port': args.triton_port}, 'keyboard', True)
    pprint(master_handler.execute(img))
