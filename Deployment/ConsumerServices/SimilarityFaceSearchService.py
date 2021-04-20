from Deployment.ConsumerWorker import celery_worker_app
from Deployment.server_config import MILVUS_URL, MILVUS_PORT
from Utils.ServiceUtils import ServiceTask
from Utils.VectorSimilarityHelpers import MilvusHelper

# 基于milvus进行特征向量检索
from Utils.VectorSimilarityHelpers.BaseVectorSimilarityHelper import VectorMetricType, VectorIndexType

similar_vector_search_helper = MilvusHelper(MILVUS_URL, MILVUS_PORT)


@celery_worker_app.task(name="ConsumerServices.SimilarityFaceSearchService.similar_face_search")
def similar_face_search(database_name, face_vector, top_k):
    search_result = similar_vector_search_helper.search(database_name, face_vector, top_k)
    return {
        'similar_ids': search_result.id_array,
        'similar_distance': search_result.distance_array,
    }


class SimilarFaceSearchServiceTask(ServiceTask):
    service_version = 'v1.0'
    service_name = 'similar_face_search'
    mock_result = {
        'similar_ids': [],
        'similar_distance': [],
    }
    require_field = {
        "database_name",
        "face_vector",
        "top_k",
    }
    binding_service = similar_face_search


@celery_worker_app.task(name="ConsumerServices.SimilarityFaceSearchService.face_insert")
def face_insert(database_name, face_vectors):
    if not similar_vector_search_helper.database_exist(database_name):
        similar_vector_search_helper.create_database(database_name, 512, 1024, VectorMetricType.L2)
        similar_vector_search_helper.create_index(database_name, VectorIndexType.IVF_PQ)
    ids = similar_vector_search_helper.insert(database_name, face_vectors, )
    return {
        'insert_ids': ids
    }


class FaceInsertServiceTask(ServiceTask):
    service_version = 'v1.0'
    service_name = 'face_insert'
    mock_result = {
        'insert_ids': [],
    }
    require_field = {
        "database_name",
        "face_vectors",
    }
    binding_service = face_insert


@celery_worker_app.task(name="ConsumerServices.SimilarityFaceSearchService.face_delete")
def face_delete(database_name, to_delete_ids):
    is_deleted = False
    if similar_vector_search_helper.database_exist(database_name):
        is_deleted = similar_vector_search_helper.delete(database_name, to_delete_ids, )
    return {
        'is_deleted': is_deleted
    }


class FaceDeleteServiceTask(ServiceTask):
    service_version = 'v1.0'
    service_name = 'face_delete'
    mock_result = {
        'is_deleted': True,
    }
    require_field = {
        "database_name",
        "to_delete_ids",
    }
    binding_service = face_insert
