from collections import defaultdict
from random import randint
from typing import Dict, List

import numpy as np


def random(**kwargs) -> Dict[str, List[int]]:
    """
    each kwargs.params=(min,max,num_value)
    """
    hyper_params = defaultdict(lambda: [])
    for args, sizes in kwargs.items():
        hyper_params[args] = [int(n) for n in np.linspace(sizes[0], sizes[1], sizes[2])]
    return hyper_params


def build(hyper_params: Dict[str, List[int]], num_runs=0):
    """
    example: build({'epoch': [1,25], 'batch': [999,50], 'round': [10,200]})
    """
    max_runs = calculate_max_rounds(hyper_params)
    if num_runs == 0:
        num_runs = max_runs

    if num_runs > max_runs:
        raise Exception(f'cant generate more than {max_runs} runs')

    runs = 0
    generated_params = []
    while runs < num_runs:
        params = {}
        for param, value in hyper_params.items():
            params[param] = value[randint(0, len(value) - 1)]
        if params in generated_params:
            continue

        generated_params.append(params)
        runs += 1
    return generated_params

def calculate_max_rounds(hyper_params: Dict[str, List[int]]):
    max_rounds = 1
    for param, value in hyper_params.items():
        max_rounds *= len(value)
    return max_rounds
