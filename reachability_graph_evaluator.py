from collections import namedtuple
from functools import reduce
from itertools import chain
from typing import Any, Callable, Dict, Set, Tuple

from network import Network, Vulnerability
from reachability_graph import Variable, External, MinAnd, MaxOr, ReachabilityGraph, linear_filter


Subgraph = namedtuple('Subgraph', 'inbound_port outbound_port variables')


def node_subgraph(node: Network.Node, vulnerabilities: Set[Tuple[Variable, float]]):

    variables: Set[Variable] = set()

    def var(**kwargs) -> Variable:
        v = Variable(**kwargs)
        variables.add(v)
        return v

    if not node.lower_nodes:
        inbound_port = None
        outbound_port = var(name=node.name, dependency=External(reachability=1.0))
    else:
        inbound_port = var(name=f'{node.name}_in')  # corresponds to "In".
        outbound_port = var(name=f'{node.name}_out')  # corresponds to "Out".

        zeroday_vulnerability = var(name=f'{node.name}_ex_v',
                                    dependency=External(reachability=1.0))
        zeroday_exploit = var(name=f'{node.name}_ex',
                              dependency=MinAnd(filter=linear_filter((100 - node.inherent_rigidity) / 100),
                                                parents={zeroday_vulnerability, inbound_port}))

        def exploits_of(vulnerabilities: Set[Tuple[Variable, float]], entrance: Variable) -> Set[Variable]:
            return {var(name=f'E_{node.name}_{vul.name}',
                        dependency=MinAnd(filter=linear_filter((100 - rigidity) / 100), parents={entrance, vul}))  # determine a_x for each filter function coupled with an AND node.
                    for vul, rigidity in vulnerabilities}

        if not vulnerabilities:
            outbound_port.dependency = MaxOr(filter=linear_filter(1), parents={zeroday_exploit})

        elif len(vulnerabilities) <= 2:
            outbound_port.dependency = MaxOr(filter=linear_filter(1), parents={zeroday_exploit, *exploits_of(vulnerabilities, inbound_port)})

        else:
            lower_vulnerabilities = set(sorted(vulnerabilities, key=lambda v: v[0].name)[:round(len(vulnerabilities) / 2)])
            upper_vulnerabilities = vulnerabilities - lower_vulnerabilities

            junction = var(name=f'J_{node.name}', dependency=MaxOr(filter=linear_filter(1), parents=exploits_of(lower_vulnerabilities, inbound_port)))
            outbound_port.dependency = MaxOr(filter=linear_filter(1), parents={zeroday_exploit, *exploits_of(upper_vulnerabilities, junction)})

    return Subgraph(inbound_port, outbound_port, variables)


class ReachabilityGraphEvaluator:

    def __init__(self, nw: Network):

        bridges: Set[Variable] = set()
        self.bno = 0
        def Bridge(source: Variable) -> Variable:
            self.bno += 1
            var = Variable(name=f'B_{self.bno}', dependency=MinAnd(parents={source}, filter=linear_filter(1)))
            bridges.add(var)
            return var

        vul_vars = {v: (Variable(name=v.name, dependency=External(reachability=1.0)), v.rigidity) for v in nw.vulnerabilities}
        subgraphs = {n: node_subgraph(n, {vul_vars[v] for v in n.vulnerabilities}) for n in nw.all_nodes}

        for node in chain(nw.clients, nw.gateways, nw.servers, nw.databases):
            subgraphs[node].inbound_port.dependency = MaxOr(parents={Bridge(subgraphs[inbound].outbound_port) for inbound in node.lower_nodes}, filter=linear_filter(1))

        self.rg = ReachabilityGraph(reduce(lambda a, b: a | b, (g.variables for g in subgraphs.values()), {v for v, r in vul_vars.values()}) | bridges)

    def evaluate(self) -> Dict[str, float]:
        return {v.name: r for v, r in self.rg.query().items()}


