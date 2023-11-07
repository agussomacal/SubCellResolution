import psutil

RAM = psutil.virtual_memory().total / 1000000000  # in GB
if RAM > 70:  # if run in server
    print("running in server")
    N = int(1e6)
    num_cores = workers = 50
elif RAM > 40:
    print("running in local power machine")
    N = int(1e5)
    num_cores = workers = 15
else:
    print("running in local machine")
    N = int(1e4)
    num_cores = workers = 1

recalculate = False
