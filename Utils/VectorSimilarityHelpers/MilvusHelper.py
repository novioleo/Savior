from Utils.VectorSimilarityHelpers.BaseVectorSimilarityHelper import BaseVectorSimilarityHelper


class MilvusHelper(BaseVectorSimilarityHelper):
    def __init__(self, _server_url, _server_port, _database_name):
        super().__init__(_server_url, _server_port, _database_name)