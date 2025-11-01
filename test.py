from tasks_PPI import celery as celery_app

from celery.result import AsyncResult

task_id = "466ffe38-bfde-46dd-899a-e494e00e8235"
res = AsyncResult(task_id, app=celery_app)
print(res.state)
print(res.result)