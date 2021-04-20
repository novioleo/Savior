from milvus import Milvus
from milvus.client.types import MetricType, IndexType

from Utils.Exceptions import MilvusRuntimeException, DatabaseNotExist
from Utils.VectorSimilarityHelpers.BaseVectorSimilarityHelper import BaseVectorSimilarityHelper, VectorMetricType, \
    VectorIndexType


class MilvusHelper(BaseVectorSimilarityHelper):
    def __init__(self, _server_url, _server_port, _timeout=10):
        super().__init__()
        self.server_url = _server_url
        self.server_port = _server_port
        self.timeout = _timeout
        self.client = None
        self.metric_type_mapper = {
            VectorMetricType.L2: MetricType.L2,
            VectorMetricType.IP: MetricType.IP,
            VectorMetricType.JACCARD: MetricType.JACCARD,
            VectorMetricType.HAMMING: MetricType.HAMMING,
        }

        self.index_type_mapper = {
            VectorIndexType.FLAT: IndexType.FLAT,
            VectorIndexType.IVFLAT: IndexType.IVFLAT,
            VectorIndexType.IVF_SQ8: IndexType.IVF_SQ8,
            VectorIndexType.RNSG: IndexType.RNSG,
            VectorIndexType.IVF_SQ8H: IndexType.IVF_SQ8H,
            VectorIndexType.IVF_PQ: IndexType.IVF_PQ,
            VectorIndexType.HNSW: IndexType.HNSW,
            VectorIndexType.ANNOY: IndexType.ANNOY,
        }

    def init(self):
        if self.client is None:
            if not (self.server_url is None or self.server_url is None):
                try:
                    self.client = Milvus(host=self.server_url, port=self.server_port)
                except:
                    raise MilvusRuntimeException(f'cannot connect to {self.server_url}:{self.server_port}')
            else:
                raise MilvusRuntimeException('Milvus config is not correct')

    def insert(self, _database_name, _to_insert_vector, _partition_tag=None, _params=None):
        """
        向数据库中插入一系列的特征向量

        notes：如果用户有自己的id，建议使用insert_with_id函数

        ATTENTION!!!

        一个库中不能既调用insert_with_id还调用insert，只能调用一种，否则会报错

        Args:
            _database_name:     数据库名称
            _to_insert_vector:  待插入的特征向量的列表
            _partition_tag:     分区标签
            _params:    插入参数

        Returns:    插入后的id

        """
        self.init()
        status, ids = self.client.insert(_database_name, _to_insert_vector, partition_tag=_partition_tag,
                                         params=_params,
                                         timeout=self.timeout)
        self.flush(_database_name)
        if status.OK():
            return ids
        else:
            raise MilvusRuntimeException(status.message)

    def insert_with_id(self, _database_name, _to_insert_vector, _to_insert_ids, _partition_tag=None, _params=None):
        """
        向数据库中插入一系列的有固定id的特征向量

        ATTENTION!!!

        一个库中不能既调用insert_with_id还调用insert，只能调用一种，否则会报错

        Args:
            _database_name:     数据库名称
            _to_insert_vector:  待插入的特征向量的列表
            _to_insert_ids:  待插入的特征向量的id的列表，每个元素必须为正整数，且不越界
            _partition_tag:     分区标签
            _params:    插入参数

        Returns:    插入后的id

        """
        self.init()
        status, ids = self.client.insert(_database_name, _to_insert_vector,
                                         ids=_to_insert_ids, partition_tag=_partition_tag,
                                         params=_params,
                                         timeout=self.timeout)
        self.flush(_database_name)
        if status.OK():
            return ids
        else:
            raise MilvusRuntimeException(status.message)

    def delete(self, _database_name, _to_delete_ids):
        """
        删除特定id

        Args:
            _database_name:     数据库名称
            _to_delete_ids:     待删除的id

        Returns:    是否删除成功

        """
        self.init()
        status = self.client.delete_entity_by_id(_database_name, _to_delete_ids, self.timeout)
        self.flush(_database_name)
        if status.OK():
            return True
        else:
            raise MilvusRuntimeException(status.message)

    def database_exist(self, _database_name):
        """
        数据库是否存在

        Args:
            _database_name:     数据库名称

        Returns:    是否存在

        """
        self.init()
        status, is_exist = self.client.has_collection(_database_name, self.timeout)
        if status.OK():
            return is_exist
        else:
            raise MilvusRuntimeException(status.message)

    def create_database(self, _database_name, _dimension, _index_file_size, _metric_type):
        """
        创建数据库

        Args:
            _database_name:     数据库名称
            _dimension:     特征向量维度
            _index_file_size:   index的文件大小
            _metric_type:   度量类型

        Returns:    是否创建成功

        """
        self.init()
        if not self.database_exist(_database_name):
            assert _metric_type in self.metric_type_mapper, f'{_metric_type} not support in milvus'
            status = self.client.create_collection({
                'collection_name': _database_name,
                'dimension': _dimension,
                'index_file_size': _index_file_size,
                'metric_type': self.metric_type_mapper[_metric_type]
            })
            if status.OK():
                return True
            else:
                raise MilvusRuntimeException(status.message)
        else:
            return True

    def create_index(self, _database_name, _index_type):
        """
        创建index（索引）

        Args:
            _database_name:     数据库名称
            _index_type:    index类型

        Returns:    是否创建成功

        """
        self.init()
        if self.database_exist(_database_name):
            assert _index_type in self.index_type_mapper, f'{_index_type} not support in milvus'
            status = self.client.create_index(_database_name, self.index_type_mapper[_index_type], timeout=self.timeout)
            if status.OK():
                return True
            else:
                raise MilvusRuntimeException(status.message)

    def search(self, _database_name, _query_vector_list, _top_k, _partition_tag=None, _params=None):
        """
        检索用参数

        Args:
            _database_name:     数据库名称
            _query_vector_list:      检索用的特征向量列表
            _top_k:     top k
            _partition_tag:     分区标签
            _params:    检索参数

        Returns:    检索的结果，包含id和distance

        """
        self.init()
        if self.database_exist(_database_name):
            status, search_result = self.client.search(_database_name, _top_k, _query_vector_list,
                                                       partition_tags=_partition_tag,
                                                       params=_params,
                                                       timeout=self.timeout)
            if status.OK():
                return search_result
            else:
                raise MilvusRuntimeException(status.message)
        else:
            raise DatabaseNotExist(f'{_database_name} not exist')

    def flush(self, _database_name):
        """
        sink数据库

        Args:
            _database_name:     数据库名称

        Returns:    是否flush成功

        """
        self.init()
        status = self.client.flush([_database_name, ], self.timeout)
        if status.OK():
            return True
        else:
            raise MilvusRuntimeException(status.message)

    def __del__(self):
        if self.client is not None:
            self.client.close()
