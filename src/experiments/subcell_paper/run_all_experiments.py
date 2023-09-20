import os
import subprocess
from time import time

os.chdir("./../../")
print(os.getcwd())
dtime = dict()
for experiment_name in ["ex_aero", "ex_binary_images", "ex_refinement", "ex_regular", "ex_scheme"]:
    try:
        print("========= ========= ========= ========= =========")
        print(f"         Running {experiment_name}              ")
        print("========= ========= ========= ========= =========")
        t0 = time()
        process = subprocess.Popen("python experiments/subcell_paper/" + experiment_name + ".py", shell=True)
        process.wait()
        dtime[experiment_name] = time() - t0
    except:
        print("Failed to run ", experiment_name)

print(dtime)
