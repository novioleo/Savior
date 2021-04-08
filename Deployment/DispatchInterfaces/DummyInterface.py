from fastapi import APIRouter, Form
from fastapi.responses import ORJSONResponse

from Deployment.ConsumerServices.DummyService import DummyService4Task, DummyService1Task, DummyService2Task, \
    DummyService3Task
from Utils.DAG import DAG
from Utils.ServiceUtils import wait_and_compose_all_task_result

router = APIRouter()


@router.post('/dummy_interface_1')
async def dummy_interface(
        dummy_input_1: str = Form(...),
        dummy_input_2: int = Form(...),
        dummy_input_3: float = Form(...),
):
    dag = DAG()
    task1 = DummyService1Task(_dag=dag)
    task1.add_dependency_from_value('dummy_input_1', dummy_input_1)
    task2 = DummyService2Task(_dag=dag)
    task2.add_dependency_from_value('dummy_input_1', dummy_input_2)
    task3 = DummyService3Task(_dag=dag)
    task3.add_dependency_from_task('dummy_input_1', task2, 'result_1')
    task4 = DummyService4Task(_dag=dag)
    task4.add_dependency_from_value('dummy_input_1', dummy_input_3)
    task4.add_dependency_from_task('dummy_input_2', task3, 'result_1')
    task4.add_dependency_from_task('dummy_input_3', task3, 'result_2')
    task4.add_dependency_from_task('dummy_input_4', task2, 'result_1')
    task4.add_dependency_from_task('dummy_input_5', task1, 'result_1')
    final_result = await wait_and_compose_all_task_result(task1, task2, task3, task4)
    to_return_result = dict()
    to_return_result['result'] = final_result
    to_return_result['dag'] = dag
    return ORJSONResponse(to_return_result)
