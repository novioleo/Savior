from fastapi import APIRouter, Form
from fastapi.responses import ORJSONResponse

from Deployment.ConsumerServices.DummyService import DummyService4Task, DummyService1Task, DummyService2Task, \
    DummyService3Task

router = APIRouter()


@router.post('/dummy_interface_1')
async def dummy_interface(
        dummy_input_1: str = Form(...),
        dummy_input_2: int = Form(...),
        dummy_input_3: float = Form(...),
):
    task1 = DummyService1Task()
    task1.add_dependency_from_value('dummy_input_1', dummy_input_1)
    task2 = DummyService2Task()
    task2.add_dependency_from_value('dummy_input_1', dummy_input_2)
    task3 = DummyService3Task()
    task3.add_dependency_from_task('dummy_input_1', task2, 'result_1')
    task4 = DummyService4Task()
    task4.add_dependency_from_value('dummy_input_1', dummy_input_3)
    task4.add_dependency_from_task('dummy_input_2', task3, 'result_1')
    task4.add_dependency_from_task('dummy_input_3', task3, 'result_2')
    task4.add_dependency_from_task('dummy_input_4', task2, 'result_1')
    task4.add_dependency_from_task('dummy_input_5', task1, 'result_1')
    final_result = await DummyService1Task.wait_and_compose_all_task_result(task1, task2, task3, task4)
    return ORJSONResponse(final_result)
