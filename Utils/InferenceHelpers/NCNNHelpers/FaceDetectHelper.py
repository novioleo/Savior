from Utils.Exceptions import NetworkInputParameterException, NetworkInitFailException
from Utils.InferenceHelpers.BaseInferenceHelper import NCNNInferenceHelper


class FaceDetectInferenceHelper(NCNNInferenceHelper):

    def __init__(self):
        super().__init__('Face Detect')
        try:
            from Libraries.face_detect import FaceDetectInfer
            self.handler = FaceDetectInfer(None, None)
        except Exception as e:
            raise NetworkInitFailException('Face Detect network init fail')
        self.add_input('image', (224, 224, 3), '将原始照片进行center pad之后resize至224')
        self.add_output('bboxes', (-1, 5), '经过NMS之后的人脸的bbox，'
                                           '5个维度分别为left top x,left top y,right bottom x,right bottom y,score')

    def infer(self, _image):
        check_status, status_message = self.all_inputs[0].tensor_check(_image)
        if check_status:
            bboxes = self.handler.get_infer_result(_image)
            return bboxes
        else:
            raise NetworkInputParameterException(status_message)
