from Deployment.ConsumerWorker import celery_worker_app
from Utils.ServiceUtils import ServiceTask


@celery_worker_app.task(name="ConsumerServices.DummyService.dummy_service_1")
def dummy_service_1(dummy_input_1):
    return {
        'result_1': dummy_input_1
    }


class DummyService1Task(ServiceTask):
    service_version = 'v1.0'
    service_name = 'dummy_service1'
    mock_result = {
        'result_1': 'service1 dummy result1',
    }
    require_field = {
        "dummy_input_1",
    }
    binding_service = dummy_service_1


@celery_worker_app.task(name="ConsumerServices.DummyService.dummy_service_2")
def dummy_service_2(dummy_input_1):
    return {
        'result_1': f'{dummy_input_1} result1',
        'result_2': f'{dummy_input_1} result2',
    }


class DummyService2Task(ServiceTask):
    service_version = 'v1.0'
    service_name = 'dummy_service2'
    mock_result = {
        'result_1': 'service2 dummy result1',
        'result_2': 'service2 dummy result2',
    }
    require_field = {
        "dummy_input_1",
    }
    binding_service = dummy_service_2


@celery_worker_app.task(name="ConsumerServices.DummyService.dummy_service_3")
def dummy_service_3(dummy_input_1):
    return {
        'result_1': f'service3 dummy result1:{dummy_input_1}',
        'result_2': f'service3 dummy result2:{dummy_input_1}',
    }


class DummyService3Task(ServiceTask):
    service_version = 'v1.0'
    service_name = 'dummy_service3'
    mock_result = {
        'result_1': 'service3 dummy result1',
        'result_2': 'service3 dummy result2',
    }
    require_field = {
        "dummy_input_1",
    }
    binding_service = dummy_service_3


@celery_worker_app.task(name="ConsumerServices.DummyService.dummy_service_4")
def dummy_service_4(dummy_input_1, dummy_input_2, dummy_input_3, dummy_input_4, dummy_input_5):
    return {
        'result': f'service_4 result:\n'
                  f'{dummy_input_1},\n'
                  f'{dummy_input_2},\n'
                  f'{dummy_input_3},\n'
                  f'{dummy_input_4},\n'
                  f'{dummy_input_5},\n',
    }


class DummyService4Task(ServiceTask):
    service_version = 'v1.0'
    service_name = 'dummy_service4'
    mock_result = {
        'result': 'service4 dummy result',
    }
    require_field = {
        "dummy_input_1",
        "dummy_input_2",
        "dummy_input_3",
        "dummy_input_4",
        "dummy_input_5",
    }
    binding_service = dummy_service_4
