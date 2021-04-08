from celery import Celery
from kombu import Queue

from Deployment.server_config import WORKER_RABBITMQ_USERNAME, WORKER_RABBITMQ_PASSWORD, WORKER_RABBITMQ_HOST, \
    WORKER_RABBITMQ_PORT, WORKER_RABBITMQ_VHOST, TASK_QUEUE, AVAILABLE_SERVICES

celery_worker_app = Celery(
    "algorithm_worker",
    backend="rpc://",
    broker=f"amqp://{WORKER_RABBITMQ_USERNAME}:{WORKER_RABBITMQ_PASSWORD}@"
           f"{WORKER_RABBITMQ_HOST}:{WORKER_RABBITMQ_PORT}/{WORKER_RABBITMQ_VHOST}",
)

celery_worker_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    enable_utc=True,
    timezone='Asia/Shanghai'
)

celery_worker_app.conf.task_queues = (
    Queue(TASK_QUEUE, routing_key='ConsumerServices.#'),
)

# 配置所需要的service的package name
celery_worker_app.autodiscover_tasks(AVAILABLE_SERVICES, related_name=None, force=True)
