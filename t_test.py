import csv
import math
import os
import numpy as np
from scipy import stats

def file_to_mean_and_var(file_path: str) -> (float, float):
    numeric_data = []
    first_line = True
    with open(file_path, 'r') as data:
        reader = csv.reader(data)
        for row in reader:
            if first_line:
                first_line = False
                continue
            numeric_data.append(float(row[1]))

    return np.mean(numeric_data), np.var(numeric_data), len(numeric_data)


def independent_sample_t_test(
        mean_1: float, var_1: float, n_1: int,
        mean_2: float, var_2: float, n_2: int,
        p_value: float) -> dict:
    pooled_variance = ((n_1 - 1) * var_1 + (n_2 - 1) * var_2)/(n_1 + n_2 - 2)
    t_statistic = (mean_1 - mean_2)/math.sqrt(pooled_variance*(1/n_1 + 1/n_2))
    t_critical_value = stats.t.ppf(1-0.5*p_value, n_1+n_2-2)
    reject_null = True if abs(t_statistic) > t_critical_value else False
    pval = stats.t.sf(np.abs(t_statistic), n_1+n_2-2)*2
    return {"t_statistic": t_statistic, "t_critical_value": t_critical_value, "reject_null": reject_null, "p_value": pval}

enemies = [3, 6, 8]

for enemy in enemies:
    normal_controller_path = f"neat_experiment/results/enemy_{enemy}/testing_results/normal_controller/result.csv"
    memory_controller_path = f"neat_experiment/results/enemy_{enemy}/testing_results/memory_controller/result.csv"

    normal_controller_mean, normal_controller_var, n_normal = file_to_mean_and_var(file_path=normal_controller_path)
    memory_controller_mean, memory_controller_var, n_memory = file_to_mean_and_var(file_path=memory_controller_path)

    print("Testing for H_0: mean_normal = mean_memory, H_a: mean_normal =\= mean_memory with significance level 0.05")
    result = independent_sample_t_test(mean_1=normal_controller_mean, var_1=normal_controller_var, n_1=n_normal,
                                      mean_2=memory_controller_mean, var_2=memory_controller_var, n_2=n_memory,
                                      p_value=0.05)
    if result["reject_null"]:
        print("Rejected null hypothesis in favor of alternative hypothesis. Evidence the normal controller and memory controller have different performance")
    else:
        print("Could not reject null hypothesis, no evidence controllers have different performance.")

    print(f"mean_normal: {normal_controller_mean}\n"
          f"var_normal: {normal_controller_var}\n"
          f"n_normal: {n_normal}")
    print(f"mean_memory: {memory_controller_mean}\n"
          f"var_memory: {memory_controller_var}\n"
          f"n_memory: {n_memory}")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n")
    #
    # print("Testing for H_0: mean_normal = mean_memory, H_a: mean_memory > mean_normal with significance level 0.05")
    # result_memory = one_tailed_sample_t_test(mean_1=memory_controller_mean,
    #                                   var_1=memory_controller_var, n_1=n_memory,
    #                                   mean_2=normal_controller_mean,
    #                                   var_2=normal_controller_var, n_2=n_normal,
    #                                   p_value=0.05)
    # if result_memory["reject_null"]:
    #     print(
    #         "Rejected null hypothesis in favor of alternative hypothesis. Evidence "
    #         "the memory controller performs better on average.")
    # else:
    #     print(
    #         "Could not reject null hypothesis, no evidence memory controller performs better."
    #     )
    #
    # for key, value in result_memory.items():
    #     print(f"{key}: {value}")
    #
    # print("\n")


