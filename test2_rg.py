from benchmark import Benchmark, run_in_parallel
from network import Network
from bayesian_network_evaluator import BayesianNetworkEvaluator
from reachability_graph_evaluator import ReachabilityGraphEvaluator


def test2_rg(nw: Network) -> Benchmark:
    evaluator = ReachabilityGraphEvaluator(nw)
    with Benchmark('test2-rg').run(nw) as bench:
        bench.reachabilities = evaluator.evaluate()
        bench.select_count = evaluator.rg.select_count
        bench.update_count = evaluator.rg.update_count
        bench.children_count = evaluator.rg.children_count
        bench.avg_queue_len = evaluator.rg.avg_queue_len
        bench.max_queue_len = evaluator.rg.max_queue_len
        return bench


def run_test2_rg(block_head: int = 1, block_size: int = 50):
    for nw_size in ['nw1', 'nw2', 'nw3', 'nw4', 'nw5']:
        for nw in Network.load_group(nw_size, block_head, block_size):
            test2_rg(nw).save()


if __name__ == "__main__":
    run_in_parallel(__file__, run_test2_rg)
