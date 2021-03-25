from abc import ABC
import numpy as np

from Operators.DummyAlgorithm import DummyAlgorithm


class FeatureVectorDistance(DummyAlgorithm, ABC):
    name = '特征向量距离度量'
    __version__ = 'v1.0.20210325'

    def feature_vector_check(self, _vector_1, _vector_2):
        assert _vector_1.ndim == _vector_2.ndim == 1, 'two vector must be 1 D'


class EuclideanDistance(FeatureVectorDistance):
    name = '欧式距离'
    __version__ = 'v1.0.20210325'

    def execute(self, _feature_vector_1, _feature_vector_2):
        self.feature_vector_check(_feature_vector_1, _feature_vector_2)
        return np.linalg.norm(np.array(_feature_vector_1)-np.array(_feature_vector_2))


class CosineDistance(FeatureVectorDistance):
    name = '余弦距离'
    __version__ = 'v1.0.20210325'

    def execute(self, _feature_vector_1, _feature_vector_2):
        self.feature_vector_check(_feature_vector_1, _feature_vector_2)
        vector_1 = np.array(_feature_vector_1)
        vector_1_norm = np.linalg.norm(vector_1)
        vector_2 = np.array(_feature_vector_2)
        vector_2_norm = np.linalg.norm(vector_2)
        return 1 - (vector_1 @ vector_2.T) / (vector_1_norm * vector_2_norm)


if __name__ == '__main__':
    feature_vector_1 = np.random.uniform(10, 20, (10, ))
    feature_vector_2 = feature_vector_1 + 1
    euclidean_distance_handler = EuclideanDistance(False)
    cosine_distance_handler = CosineDistance(False)
    print(euclidean_distance_handler.execute(feature_vector_1,feature_vector_2))
    print(cosine_distance_handler.execute(feature_vector_1,feature_vector_2))
