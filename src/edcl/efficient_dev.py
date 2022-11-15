import edcl as di
import numpy as np
from math import ceil
from time import perf_counter_ns
from scipy.stats import percentileofscore

test_list = [100, 100, 100, 100]


def percentile_from_ordered(ordered, percentile):
    idx = (percentile / 100) * (len(ordered)-1)
    return ordered[ceil(idx)]


def fraction_below_ordered(ordered, score):
    quan_less = 0
    for x in ordered:
        if x < score:
            quan_less += 1
        else:
            break
    return (quan_less / len(ordered)) * 100


my_start = perf_counter_ns()
print(fraction_below_ordered(test_list, 100))
my_end = perf_counter_ns()

np_start = perf_counter_ns()
print(percentileofscore(test_list, 100))
np_end = perf_counter_ns()

print((np_end - np_start) / (my_end - my_start))
