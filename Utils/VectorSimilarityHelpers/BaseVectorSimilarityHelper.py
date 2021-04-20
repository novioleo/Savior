from enum import IntEnum


class VectorMetricType(IntEnum):
    L2: int = 0
    IP: int = 1
    HAMMING: int = 2
    JACCARD: int = 3


class VectorIndexType(IntEnum):
    FLAT = 0
    IVFLAT = 1
    IVF_SQ8 = 2
    RNSG = 3
    IVF_SQ8H = 4
    IVF_PQ = 5
    HNSW = 6
    ANNOY = 7


class BaseVectorSimilarityHelper:
    def __init__(self):
        pass

    def search(self, _database_name, _query_vector, _top_k):
        pass

    def insert(self, _database_name, _to_insert_vector):
        pass

    def insert_with_id(self,_database_name,_to_insert_vector,_to_insert_ids, _partition_tag=None, _params=None):
        pass

    def delete(self, _database_name, _to_delete_vector_id):
        pass

    def database_exist(self, _database_name):
        pass

    def create_database(self, _database_name, _dimension, _index_file_size, _metric_type):
        pass

    def drop_database(self, _database_name):
        pass

    def create_partition(self, _database_name, _partition_name):
        pass

    def drop_partition(self, _database_name, _partition_name):
        pass

    def create_index(self, _database_name, _index_type: VectorIndexType):
        pass

    def drop_index(self, _database_name):
        pass
