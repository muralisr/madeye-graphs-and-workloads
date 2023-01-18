import random
from datetime import datetime
random.seed(100)

class Query():
    def __init__(self):
        self.query = ""
        self.object = ""
        self.model = ""

    def __str__(self) -> str:
        return f'(\'{self.model}\', \'{self.query}\', \'{self.object}\')'

    def __repr__(self) -> str:
        return f'(\'{self.model}\', \'{self.query}\', \'{self.object}\')'

def get_random_model():
    options_for_model = ["tiny-yolov4", "yolov4", "faster-rcnn", "ssd"]
    index = random.randint(0, len(options_for_model) - 1)
    return options_for_model[index]

def get_random_object():
    options_for_object = ["person", "car"]
    index = random.randint(0, len(options_for_object) - 1)
    return options_for_object[index]

def get_random_query():
    options_for_query = ["count", "binary-classification", "detection", "aggregate-count"]
    index = random.randint(0, len(options_for_query) - 1)
    return options_for_query[index]

def generate_random_workload(workload_size=5):
    workload = []
    for _ in range(workload_size):
        query = get_random_query()
        if query == 'aggregate-count':
            obj = 'person'
        else:
            obj = get_random_object()
        model = get_random_model()
        q = Query()
        q.query = query
        q.object = obj
        q.model = model
        workload.append(q)
#    type = "small"
#    if workload_size >= medium_workload_size_bounds[0] and workload_size <= medium_workload_size_bounds[1]:
#        type = "medium"
#    elif workload_size >= large_workload_size_bounds[0] and workload_size <= large_workload_size_bounds[1]:
#        type = "large"    
    return (workload, None)

if __name__ == "__main__":
    num_workloads_reqd = 20
    workloads = []
    for _ in range(num_workloads_reqd):
        workloads.append(generate_random_workload(random.randint(2, 20)))

    for i in range(len(workloads)):
        print(f"{i+1}:{workloads[i]}")

    with open(f"generated workload {datetime.now()}.txt", "w") as f:
        for i in range(len(workloads)):
            f.write(f"{i+1}:{workloads[i]}\n")
