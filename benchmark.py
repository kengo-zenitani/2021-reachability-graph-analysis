from contextlib import contextmanager
from math import ceil
import os
from subprocess import Popen
from statistics import mean, stdev
import sys
import time
from typing import List, Tuple, Callable

from common import Problem, load_object, save_object


PYTHON_PATH = 'python3.7'
PYPY_PATH = 'pypy'


@contextmanager
def timeit():
    class Watch:
        def __init__(self, start: float):
            self.start: float = start
            self.end: float = 0
            self.time: float = 0

    watch = Watch(start=time.perf_counter())
    try:
        yield watch
    finally:
        watch.end = time.perf_counter()
        watch.time = watch.end - watch.start


class Benchmark:
    def __init__(self, algorithm_tag: str, problem_tag: str = ''):
        self.algorithm_tag: str = algorithm_tag
        self.problem_tag: str = problem_tag
        self.processing_time: float = 0

    @contextmanager
    def run(self, pb: Problem) -> 'Benchmark':
        self.problem_tag = pb.name

        print(f"{self.algorithm_tag}-{self.problem_tag}: start ...")
        self.processing_time = 0

        with timeit() as watch:
            yield self

        self.processing_time += watch.time
        print(f"{self.algorithm_tag}-{self.problem_tag}: complete.")

        self.report()

    def preprocess(self, preprocess: 'Benchmark') -> 'Benchmark':
        self.processing_time += preprocess.processing_time
        return self

    def report(self) -> 'Benchmark':
        print(f'{self.algorithm_tag}-{self.problem_tag}: processing time={self.processing_time}')
        return self

    def save(self) -> 'Benchmark':
        return save_object(f'benchmark/{self.algorithm_tag}/{self.algorithm_tag}-{self.problem_tag}.bin', self)

    @classmethod
    def load(cls, algorithm_tag: str, problem_tag: str) -> 'Benchmark':
        return load_object(f'benchmark/{algorithm_tag}/{algorithm_tag}-{problem_tag}.bin')


def run_in_parallel(script_file_path: str, test_function: Callable, use_pypy: bool = False):
    if len(sys.argv) == 2:
        block_count = int(sys.argv[1])
        block_size = ceil(50 / block_count)
        for proc in [Popen(f"{PYPY_PATH if use_pypy else PYTHON_PATH} {os.path.basename(script_file_path)} {i * block_size + 1} {block_size}", shell=True) for i in range(block_count)]:
            proc.wait()
        print('complete')
    elif len(sys.argv) == 3:
        print("start -- block=" + sys.argv[1])
        test_function(int(sys.argv[1]), min(int(sys.argv[2]), 51 - int(sys.argv[1])))
        print("complete -- block=" + sys.argv[1])
    else:
        print("ERROR: arguments count must be 2 or 3.")


def load_performance_set(algorithm_name: str, problem_class: str):
    performance_set = []
    for index in range(0, 50):
        bench = Benchmark.load(algorithm_tag=algorithm_name, problem_tag=f"{problem_class}-{index + 1}")
        performance_set.append((
            bench.processing_time,
            getattr(bench, 'select_count', 0),
            getattr(bench, 'update_count', 0),
            getattr(bench, 'children_count', 0),
            getattr(bench, 'avg_queue_len', 0),
            getattr(bench, 'max_queue_len', 0)))
    return performance_set

def check_processing_times(algorithm_name: str, problem_class: str):
    processing_times = [p[0] for p in load_performance_set(algorithm_name, problem_class)]
    return mean(processing_times), stdev(processing_times), max(processing_times), min(processing_times)


def compare_processing_times(improved_algorithm_name: str, base_algorithm_name: str, problem_class: str):
    improvements = []
    improved_count = 0
    for index in range(0, 50):
        improved_result = Benchmark.load(algorithm_tag=improved_algorithm_name, problem_tag=f"{problem_class}-{index + 1}")
        base_result = Benchmark.load(algorithm_tag=base_algorithm_name, problem_tag=f"{problem_class}-{index + 1}")
        improvements.append(base_result.processing_time / improved_result.processing_time)
        improved_count += 1 if improved_result.processing_time < base_result.processing_time else 0
    return mean(improvements), stdev(improvements), max(improvements), min(improvements), improved_count, improved_count / len(improvements)


