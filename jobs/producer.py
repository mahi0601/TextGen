from rq import Queue
from redis import Redis
from app.inference import predict

redis_conn = Redis()
queue = Queue(connection=redis_conn)

def enqueue_prompt(prompt):
    job = queue.enqueue(predict, prompt)
    return job.get_id()
