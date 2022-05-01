from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from functools import reduce
from typing import Any, Callable, Dict, Set, Optional


VARIABLE_ID = 0


class Variable:
    def __init__(self, dependency: Optional['Dependency'] = None, name: str = ''):
        global VARIABLE_ID
        VARIABLE_ID += 1
        self.id = VARIABLE_ID

        self.dependency: Optional[Dependency] = dependency
        self.name: str = name

    @property
    def parents(self) -> Set['Variable']:
        return self.dependency.parents

    def query(self, assignments: Dict['Variable', float]) -> float:
        return self.dependency.query(self, assignments)

    def __hash__(self):
        return self.id.__hash__()

    def __repr__(self) -> str:
        if self.dependency.parents:
            return f"<Variable:{self.name} depends On {', '.join(p.name for p in self.parents)}>"
        else:
            return f"<Variable:{self.name}>"


class Dependency(ABC):
    parents: Set[Variable] = set()
    shape: str = 'shape = circle'

    @abstractmethod
    def query(self, subject: Variable, assignments: Dict[Variable, float]) -> float:
        pass


class External(Dependency):
    shape: str = 'shape = circle'

    def __init__(self, reachability: float = 0):
        self.reachability: float = reachability

    def query(self, subject: Variable, assignments: Dict[Variable, float]) -> float:
        return assignments.get(subject, self.reachability)


class MinAnd(Dependency):
    shape: str = 'shape = Mcircle'

    def __init__(self, parents: Set[Variable], filter: Callable[[float], float]):
        self.parents: Set[Variable] = parents.copy()
        self.filter: Callable[[float], float] = filter

    def query(self, subject: Variable, assignments: Dict[Variable, float]) -> float:
        return self.filter(min(assignments[var] for var in self.parents))


class MaxOr(Dependency):
    shape: str = 'shape = circle, style = dashed'

    def __init__(self, parents: Set[Variable], filter: Callable[[float], float]):
        self.parents: Set[Variable] = parents.copy()
        self.filter: Callable[[float], float] = filter

    def query(self, subject: Variable, assignments: Dict[Variable, float]) -> float:
        return self.filter(max(assignments[var] for var in self.parents))


class ReachabilityGraph:
    def __init__(self, variables: Optional[Set[Variable]] = None, verify: bool = False):
        self.variables: Set[Variable] = variables or set()
        if variables and verify:
            self.verify()
        self.select_count = 0  # for performance analysis
        self.update_count = 0  # for performance analysis
        self.children_count = 0  # for performance analysis
        self.avg_queue_len = 0  # for performance analysis
        self.max_queue_len = 0  # for performance analysis

    def add(self, variable: Variable, verify: bool = True):
        self.variables.add(variable)
        if verify:
            self.verify()
        return self

    def verify(self):
        assert reduce(lambda a, b: a | b, (var.parents for var in self.variables)).issubset(self.variables)

    """
    The following function implements Algorithm 1: reachability-graph analysis in the article.
    """
    def query(self, order: str = '') -> Dict[Variable, float]:
        self.select_count = 0  # for performance analysis
        self.update_count = 0  # for performance analysis
        self.children_count = 0  # for performance analysis
        self.avg_queue_len = 0  # for performance analysis
        self.max_queue_len = 0  # for performance analysis

        # The following loop defines the function "children"
        children: Dict[Variable, Set[Variable]] = defaultdict(set)
        for variable in self.variables:
            for parent in variable.parents:
                children[parent].add(variable)

        # The following 4 lines corresponds to line 2 to 8 in Algorithm 1.
        # "unchecked_variables" corresponds to "Q" in Algorithm 1.
        reachabilities: Dict[Variable, float] = defaultdict(float)
        unchecked_variables: Set[Variable] = set()
        for var in self.variables:  # self.variables is equivalent to "E \cup C"
            if isinstance(var.dependency, External):
                reachabilities[var] = var.dependency.reachability
                unchecked_variables |= children[var]
            else:
                reachabilities[var] = 0

        if not order:
            # The following while loop corresponds to the while loop from line 9 to 17.
            while unchecked_variables:
                variable = unchecked_variables.pop()
                reachability = variable.query(assignments=reachabilities)  # This "query" call runs MaxOR.query or MaxAND.query, i.e., corresponds to line 11 to 14.
                if reachabilities[variable] != reachability:  # Corresponds to the "if" at line 15.
                    reachabilities[variable] = reachability
                    unchecked_variables |= children[variable]

                    self.update_count += 1
                    qlen = len(unchecked_variables)
                    self.avg_queue_len += qlen
                    self.max_queue_len = max(self.max_queue_len, qlen)
                self.children_count += len(variable.dependency.parents)
                self.select_count += 1

        elif order == 'layered':
            # This block is used for the validation of the claim of Theorem 8.
            # "test_order_stability.py" do the validation.
            while unchecked_variables:
                next_unchecked_variables = set()
                while unchecked_variables:
                    variable = unchecked_variables.pop()
                    reachability = variable.query(assignments=reachabilities)
                    if reachabilities[variable] != reachability:
                        reachabilities[variable] = reachability
                        next_unchecked_variables |= children[variable]
                unchecked_variables = next_unchecked_variables

        elif order == 'deque':
            # This block is used for the validation of the claim of Theorem 8.
            # "test_order_stability.py" do the validation.
            from collections import deque
            unchecked_variables = deque(unchecked_variables)  # Note that deque do not merge duplications.
            while unchecked_variables:
                variable = unchecked_variables.popleft()
                reachability = variable.query(assignments=reachabilities)
                if reachabilities[variable] != reachability:
                    reachabilities[variable] = reachability
                    unchecked_variables.extend(children[variable])
        else:
            raise NotImplementedError()

        self.avg_queue_len /= self.update_count

        return reachabilities

    def to_dot(self, reachabilities: Optional[Dict[Variable, float]] = None) -> str:
        with open('reachability_graph_template.dot', 'r') as fp:
            template = fp.read()

        reachabilities = reachabilities or {}
        node_defs = [f'"{variable.name}" [{variable.dependency.shape}{f", label = {round(reachabilities[variable.name], 5)}" if variable.name in reachabilities else ""}];' for variable in self.variables]
        edge_defs = [f'"{parent.name}" -> "{child.name}";' for child in self.variables for parent in child.parents]
        return template.replace('%Nodes%', '\n  '.join(node_defs)).replace('%Edges%', '\n  '.join(edge_defs))


@contextmanager
def reachability_graph(dictionary: Dict[str, Any]) -> ReachabilityGraph:

    keys_before = set(dictionary.keys()).copy()
    rg = ReachabilityGraph()

    def commit(dictionary: Dict[str, Any]) -> ReachabilityGraph:
        for new_key in set(dictionary.keys()) - keys_before:
            new_value = dictionary[new_key]
            if isinstance(new_value, Variable) and not new_value.name:
                new_value.name = new_key
                rg.add(new_value, verify=False)
        return rg

    yield commit

    rg.verify()


def linear_filter(ax: float) -> Callable[[float], float]:
    return lambda v: ax * v


