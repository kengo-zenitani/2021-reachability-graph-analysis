from collections import namedtuple
from contextlib import contextmanager
from functools import reduce
from itertools import chain
from typing import Any, Dict, List, Set, Tuple


from bayesian_network import BayesianNet, Variable, PriorDistribution, NoisyOrDistribution, NoisyAndDistribution
from network import Network, Vulnerability


Subgraph = namedtuple('Subgraph', 'inbound_port outbound_port variables')


def node_subgraph(node: Network.Node, vulnerabilities: Set[Variable]):

    variables: Set[Variable] = set()

    def var(**kwargs) -> Variable:
        v = Variable(**kwargs)
        variables.add(v)
        return v

    if not node.lower_nodes:
        inbound_port = None
        outbound_port = var(name=node.name, distribution=PriorDistribution(true_probability=1.0))
    else:
        inbound_port = var(name=f'{node.name}_in')  # corresponds to "In".
        outbound_port = var(name=f'{node.name}_out')  # corresponds to "Out".

        zeroday_vulnerability = var(name=f'{node.name}_ex_v',
                                    distribution=PriorDistribution(true_probability=(100 - node.inherent_rigidity) / 100))
        zeroday_exploit = var(name=f'{node.name}_ex',
                              distribution=NoisyAndDistribution(probability_table={inbound_port: (1.0, 0.0), zeroday_vulnerability: (1.0, 0.0)}))

        def exploits_of(vulnerabilities: Set[Variable], entrance: Variable) -> Set[Variable]:
            return {var(name=f'E_{node.name}_{vul.name}',
                        distribution=NoisyAndDistribution({entrance: (1.0, 0.0), vul: (1.0, 0.0)}))
                    for vul in vulnerabilities}

        if not vulnerabilities:
            outbound_port.distribution = NoisyOrDistribution(
                probability_table={zeroday_exploit: (1.0, 0.0)})

        elif len(vulnerabilities) <= 2:
            outbound_port.distribution = NoisyOrDistribution(
                probability_table={zeroday_exploit: (1.0, 0.0), **{var: (1.0, 0.0) for var in exploits_of(vulnerabilities, inbound_port)}})

        else:
            lower_vulnerabilities = set(sorted(vulnerabilities, key=lambda v: v.name)[:round(len(vulnerabilities) / 2)])
            upper_vulnerabilities = vulnerabilities - lower_vulnerabilities

            junction = var(name=f'J_{node.name}', distribution=NoisyOrDistribution(
                probability_table={var: (1.0, 0.0) for var in exploits_of(lower_vulnerabilities, inbound_port)}))

            outbound_port.distribution = NoisyOrDistribution(
                probability_table={zeroday_exploit: (1.0, 0.0), **{var: (1.0, 0.0) for var in exploits_of(upper_vulnerabilities, junction)}})

    return Subgraph(inbound_port, outbound_port, variables)


class BayesianNetworkEvaluator:

    def __init__(self, nw: Network):

        bridges: Set[Variable] = set()
        self.bno = 0
        def Bridge(source: Variable) -> Variable:
            self.bno += 1
            var = Variable(name=f'B_{self.bno}', distribution=NoisyAndDistribution(probability_table={source: (1.0, 0.0)}))
            bridges.add(var)
            return var

        vul_vars = {v: Variable(name=v.name, distribution=PriorDistribution(true_probability=(100 - v.rigidity) / 100)) for v in nw.vulnerabilities}
        subgraphs = {n: node_subgraph(n, {vul_vars[v] for v in n.vulnerabilities}) for n in nw.all_nodes}

        for node in chain(nw.clients, nw.gateways, nw.servers, nw.databases):
            subgraphs[node].inbound_port.distribution = NoisyOrDistribution(
                probability_table={Bridge(subgraphs[inbound].outbound_port): (1.0, 0.0) for inbound in node.lower_nodes})

        self.bn = BayesianNet(reduce(lambda a, b: a | b, (g.variables for g in subgraphs.values()), set(vul_vars.values())) | bridges)

    def evaluate(self) -> Dict[str, float]:
        return {v.name: r for v, r in self.bn.query_all_by_ibp({}).items()}
