from benchmark import Benchmark, run_in_parallel
from network import Network
from bayesian_network_evaluator import BayesianNetworkEvaluator
from reachability_graph_evaluator import ReachabilityGraphEvaluator


def test2_bn(nw: Network) -> Benchmark:
    evaluator = BayesianNetworkEvaluator(nw)
    with Benchmark('test2-bn').run(nw) as bench:
        bench.reachabilities = evaluator.evaluate()
        return bench


def run_test2_bn(block_head: int = 1, block_size: int = 50):
    for nw_size in ['nw1', 'nw2', 'nw3', 'nw4', 'nw5']:
        for nw in Network.load_group(nw_size, block_head, block_size):
            test2_bn(nw).save()


if __name__ == "__main__":
    run_in_parallel(__file__, run_test2_bn)
