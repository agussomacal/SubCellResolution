import psutil

RAM = psutil.virtual_memory().total / 1000000000  # in GB
server = RAM > 50
if server:  # if run in server
    print("running in server")
    N = int(1e6)
else:
    print("running in local machine")
    N = int(1e4)

recalculate = False
workers = 50
