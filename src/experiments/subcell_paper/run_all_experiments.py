from PerplexityLab.miscellaneous import timeit

for experiment_name in ["ex_aero", "ex_binary_images", "ex_refinement", "ex_regular", "ex_scheme"]:
    try:
        print("========= ========= ========= ========= =========")
        print(f"         Running {experiment_name}              ")
        print("========= ========= ========= ========= =========")
        with timeit(f"Time for {experiment_name}"):
            exec(open(experiment_name + ".py").read())
    except:
        print("Failed to run ", experiment_name)
