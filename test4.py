from benchmark import Benchmark, run_in_parallel
from network import Network
import os
from reachability_graph_evaluator import ReachabilityGraphEvaluator


def test4_rg(nw: Network) -> Benchmark:
    evaluator = ReachabilityGraphEvaluator(nw)
    with Benchmark('test4').run(nw) as bench:
        bench.reachabilities = evaluator.evaluate()
        bench.select_count = evaluator.rg.select_count
        bench.update_count = evaluator.rg.update_count
        bench.children_count = evaluator.rg.children_count
        bench.avg_queue_len = evaluator.rg.avg_queue_len
        bench.max_queue_len = evaluator.rg.max_queue_len
        return bench


def run_test4(block_head: int = 1, block_size: int = 50):
    for nw_size in reversed([f'nw10-h{h * 8}' for h in range(8)]):
        for nw in Network.load_group(nw_size, block_head, block_size):
            if os.path.exists(f'benchmark/test4/test4-{nw.name}.bin'):
                print(f'test4-{nw.name} skipped')
                continue
            test4_rg(nw).save()


if __name__ == "__main__":
    run_in_parallel(__file__, run_test4)
