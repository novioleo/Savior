from Utils.Exceptions import NetworkInputParameterException
from Utils.InferenceHelpers.BaseInferenceHelper import DummyInferenceHelper
import numpy as np


class FaceDetectInferenceHelper(DummyInferenceHelper):

    def __init__(self):
        super().__init__('Face Detect')
        self.add_input('image', (224, 224, 3), '将原始照片进行center pad之后resize至224')
        self.add_output('bboxes', (-1, 5), '经过NMS之后的人脸的bbox，'
                                           '5个维度分别为left top x,left top y,right bottom x,right bottom y,score')

    def infer(self, _image):
        check_status, status_message = self.all_inputs[0].tensor_check(_image)
        if check_status:
            bboxes = np.array([[23, 23, 96, 96, 0.984], [[11, 29, 53, 77, 0.842]]], dtype=np.float32)
            return bboxes
        else:
            raise NetworkInputParameterException(status_message)
