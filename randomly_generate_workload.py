import random

random.seed(100)

small_workload_size_bounds = (2,3)
medium_workload_size_bounds = (4,8)
large_workload_size_bounds = (9,10)

class Query():
    def __init__(self):
        self.query = ""
        self.object = ""
        self.model = ""

    def __str__(self) -> str:
        return f"{self.query}-{self.object}-{self.model}"

    def __repr__(self) -> str:
        return f"{self.query}-{self.object}-{self.model}"

def get_random_model():
    options_for_model = ["tiny-yolov4", "yolov4", "frcnn", "ssd"]
    index = random.randint(0, len(options_for_model) - 1)
    return options_for_model[index]

def get_random_object():
    options_for_object = ["person", "car"]
    index = random.randint(0, len(options_for_object) - 1)
    return options_for_object[index]

def get_random_query():
    options_for_query = ["count", "classify", "detect", "aggregate"]
    index = random.randint(0, len(options_for_query) - 1)
    return options_for_query[index]

def generate_random_workload(workload_size=5):
    workload = []
    for _ in range(workload_size):
        query = get_random_query()
        obj = get_random_object()
        model = get_random_model()
        q = Query()
        q.query = query
        q.object = obj
        q.model = model
        workload.append(q)
    type = "small"
    if workload_size >= medium_workload_size_bounds[0] and workload_size <= medium_workload_size_bounds[1]:
        type = "medium"
    elif workload_size >= large_workload_size_bounds[0] and workload_size <= large_workload_size_bounds[1]:
        type = "large"    
    return (workload, type)

if __name__ == "__main__":
    num_small_workloads = 4
    num_medium_workloads = 12
    num_large_workloads = 4
    workloads = []
    for _ in range(num_small_workloads):
        workloads.append(generate_random_workload(random.randint(small_workload_size_bounds[0], small_workload_size_bounds[1])))
    for _ in range(num_medium_workloads):
        workloads.append(generate_random_workload(random.randint(medium_workload_size_bounds[0],medium_workload_size_bounds[1])))
    for _ in range(num_large_workloads):
        workloads.append(generate_random_workload(random.randint(large_workload_size_bounds[0],large_workload_size_bounds[1])))

    for i in range(len(workloads)):
        print(f"{i+1}:{workloads[i]}")