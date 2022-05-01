from functools import reduce
from itertools import chain
import random
from typing import Any, Dict, List, Set, Tuple, Iterable, Optional, Union, Type

from common import Problem, load_object, save_object


class Vulnerability:
    def __init__(self, name: str, layer: int, rigidity: int):
        super().__init__()
        self.name: str = name
        self.layer: int = layer
        self.rigidity: int = rigidity

    def __hash__(self):
        return (self.name, self.layer).__hash__()

    def __eq__(self, opponent: 'Vulnerability'):
        return self.name == opponent.name


class Network(Problem):

    class Node:
        def __init__(self, name: str, vulnerabilities: Optional[List[Vulnerability]] = None):
            super().__init__()
            self.name: str = name
            self.value: float = 0
            self.inherent_rigidity: int = random.randint(80, 99)
            self.vulnerabilities: Union[List[Vulnerability], Set[Vulnerability]] = vulnerabilities.copy() if vulnerabilities else []
            self.lower_nodes: List[Network.Node] = []

        def __hash__(self):
            return self.name.__hash__()

        def __eq__(self, opponent: 'Network.Node'):
            return self.name == opponent.name


    def __init__(self,
                 name: str,
                 nv2: int,  # Number of vulnerabilities in each node in layer-2.
                 n0: int,  # Number of database servers in layer-0.
                 n1: int,  # Number of application servers in layer-1.
                 c12: int,  # Number of connections going from each node in layer-1 to layer-2.
                 n2: int,  # Number of gateways in layer-2.
                 c23: int,  # Number of connections going from each node in layer-2 to layer-3.
                 n3: int,  # Number of clients in layer-3.
                 hub_count: int = 0):  # Number of gateways working as a bidirectional hub.

        super().__init__(name=name)

        if (n1 * c12) % n2:
            raise ValueError('ERROR: Invalid parameters specified. (n1 * c12) % n2 != 0')

        if (n2 * c23) % n3:
            raise ValueError('ERROR: Invalid parameters specified. (n2 * c23) % n3 != 0')

        if (nv2 * n2) % n1:
            raise ValueError('ERROR: Invalid parameters specified. (nv2 * n2) % n1 != 0')

        if (nv2 * n3) % n2:
            raise ValueError('ERROR: Invalid parameters specified. (nv2 * n3) % n2 != 0')

        nv1 = nv2 * n2 // n1
        nv3 = nv2 * n3 // n2

        # Make nodes
        all_vulnerabilities = []

        def _vulnerabilities(layer: int, count: int) -> List[Vulnerability]:
            vulnerabilities = [Vulnerability(name=f'v{len(all_vulnerabilities) + k + 1}@{layer}',
                                             layer=layer,
                                             rigidity=random.randint(1, 99)) for k in range(count)]
            all_vulnerabilities.extend(vulnerabilities)
            return vulnerabilities

        databases = [Network.Node(name=f'db{k + 1}') for k in range(n0)]
        servers = [Network.Node(name=f's{k + 1}', vulnerabilities=_vulnerabilities(1, nv1)) for k in range(n1)]
        gateways = [Network.Node(name=f'gw{k + 1}', vulnerabilities=_vulnerabilities(2, nv2)) for k in range(n2)]
        clients = [Network.Node(name=f'c{k + 1}') for k in range(n3)]
        internet = Network.Node(name='TheInternet')

        shifted_clients = clients.copy()
        client_vulnerabilities = _vulnerabilities(3, nv3)
        shift = n2 // nv2
        for j in range(nv2):
            for k in range(n3 // n2):
                vulnerability = client_vulnerabilities.pop()
                for c in shifted_clients[k * n2:k * n2 + n2]:
                    c.vulnerabilities.append(vulnerability)
            shifted_clients = shifted_clients[shift:] + shifted_clients[:shift]

        # Make edges
        edges = [(db, s) for db in databases for s in servers]  # layer-0 to layer-1
        edges += [(servers[i], gateways[j]) for i, j in zip(sum([([i] * c12) for i in range(n1)], []), list(range(n2)) * (n1 * c12 // n2))]  # layer-1 to layer-2
        edges += [(gateways[i], clients[j]) for i, j in zip(sum([([i] * c23) for i in range(n2)], []), list(range(n3)) * (n2 * c23 // n3))]  # layer-2 to layer-3
        edges += [(c, internet) for c in clients]  # layer-3 to layer-4

        for upper, lower in edges:
            upper.lower_nodes.append(lower)

        if hub_count:
            for gw in random.sample(gateways, hub_count):
                for c in gw.lower_nodes:
                    if random.random() < 0.5:
                        c.lower_nodes = [gw]
                for s in servers:
                    if gw in s.lower_nodes:
                        gw.lower_nodes.append(s)

        # field assignments
        self.databases: List[Network.Node] = databases
        self.servers: List[Network.Node] = servers
        self.gateways: List[Network.Node] = gateways
        self.clients: List[Network.Node] = clients
        self.internet: Network.Node = internet
        self.edges: List[Tuple[Network.Node, Network.Node]] = edges
        self.vulnerabilities: Set[Vulnerability] = set(all_vulnerabilities)

    @property
    def all_nodes(self) -> Iterable[Node]:
        return chain(self.databases, self.servers, self.gateways, self.clients, [self.internet])

    @property
    def paths(self) -> List[List[Node]]:
        return [[db, s, gw, c, inet]
                for db in self.databases
                for s in db.lower_nodes
                for gw in s.lower_nodes
                for c in gw.lower_nodes
                for inet in c.lower_nodes]

    def to_dot(self) -> str:
        with open('network_template.dot', 'r') as fp:
            template = fp.read()

        node_defs = [f'{db.name} [shape = cylinder];' for db in self.databases]
        node_defs += [f'{s.name} [shape = box3d];' for s in self.servers]
        node_defs += [f'{gw.name} [shape = component];' for gw in self.gateways]
        node_defs += [f'{c.name} [shape = square];' for c in self.clients]
        node_defs += [f'{self.internet.name} [shape = oval];']

        edge_defs = [f'{src.name} -> {dst.name};' for src, dst in self.edges]

        return template.replace('%Nodes%', '\n  '.join(node_defs)).replace('%Edges%', '\n  '.join(edge_defs))

    def save(self):
        save_object(f'network/{self.name}.pickle', self)

    @classmethod
    def load(cls, name: str) -> 'Network':
        return load_object('network/' + name + '.pickle')

    @classmethod
    def load_group(cls, nw_size: str, group_head: int, group_size: int) -> List['Network']:
        for i in range(group_head, group_head + group_size):
            yield Network.load(f"{nw_size}-{i}")


