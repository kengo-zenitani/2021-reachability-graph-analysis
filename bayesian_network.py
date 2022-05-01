from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from functools import reduce
from itertools import product
import pickle
from typing import Any, Dict, List, Set, Tuple, Iterable, Optional, FrozenSet


VARIABLE_ID = 0


def neutral() -> float:
    return 0.5


def mul(a, b):
    return a * b


def prod(values: Iterable[float]) -> float:
    values = list(values)
    return reduce(mul, values) if values else 1.0


class Distribution(ABC):
    shape: str = 'shape = circle'

    def __init__(self, conditions: Optional[Iterable['Variable']] = None):
        super().__init__()
        self.conditions: Set[Variable] = set(conditions or [])

    def prune_condition(self, subject: 'Variable', variable: 'Variable', value: bool) -> 'Distribution':
        pass

    @abstractmethod
    def query(self, subject: 'Variable', assignments: Dict['Variable', bool]) -> float:
        raise NotImplementedError()

    @abstractmethod
    def localize(self, subject: 'Variable', assignments: Dict['Variable', bool]) -> 'Distribution':
        raise NotImplementedError()

    @abstractmethod
    def local_belief(self, subject: 'Variable', assignments: Dict['Variable', bool], parents: Dict['Variable', float]) -> float:
        raise NotImplementedError()


class Variable:
    def __init__(self, name: Optional[str] = None, distribution: Optional[Distribution] = None):
        global VARIABLE_ID
        VARIABLE_ID += 1
        self.id = VARIABLE_ID

        self.name: Optional[str] = name
        self.distribution: Optional[Distribution] = distribution

    def __eq__(self, other):
        return other.id == self.id

    def __hash__(self):
        return self.id.__hash__()

    def __repr__(self) -> str:
        if self.parents:
            return f'<{self.__class__.__name__}:{self.name} depends on {", ".join([parent.name or "None" for parent in self.parents])}>'
        else:
            return f'<{self.__class__.__name__}:{self.name}>'

    @property
    def parents(self) -> Set['Variable']:
        return self.distribution.conditions if self.distribution else set()

    def prune_parent(self, variable: 'Variable', value: bool):
        self.distribution = self.distribution.prune_condition(subject=self, variable=variable, value=value)

    @abstractmethod
    def query(self, assignments: Dict['Variable', bool]) -> float:
        return self.distribution.query(subject=self, assignments=assignments)

    def to_cpd(self) -> 'Variable':
        self.distribution = self.distribution.to_cpd(self)
        return self

    def localize(self, assignments: Dict['Variable', bool]):
        self.distribution = self.distribution.localize(subject=self, assignments=assignments)

    @abstractmethod
    def local_belief(self, assignments: Dict['Variable', bool], parents: Dict['Variable', float]) -> float:
        return self.distribution.local_belief(subject=self, assignments=assignments, parents=parents)


def children_map(variables: Set[Variable]) -> Dict[Variable, Set[Variable]]:
    children: Dict[Variable, Set[Variable]] = defaultdict(set)
    for variable in variables:
        for parent in variable.parents:
            children[parent].add(variable)
    return children


class PriorDistribution(Distribution):
    shape: str = 'shape = circle'

    def __init__(self, true_probability: float):
        super().__init__()
        assert 0.0 <= true_probability <= 1.0
        self.true_probability: float = true_probability

    def prune_condition(self, subject: Variable, variable: Variable, value: bool) -> Distribution:
        raise NotImplementedError()

    def query(self, subject: Variable, assignments: Dict[Variable, bool]) -> float:
        return self.true_probability if assignments[subject] else 1 - self.true_probability

    def localize(self, subject: Variable, assignments: Dict[Variable, bool]) -> Distribution:
        if subject in assignments:
            return PriorDistribution(true_probability=1 if assignments[subject] else 0)
        else:
            return self

    def local_belief(self, subject: Variable, assignments: Dict[Variable, bool], parents: Dict[Variable, float]) -> float:
        return self.true_probability if assignments[subject] else 1 - self.true_probability

    def to_cpd(self, subject: Variable) -> 'PriorDistribution':
        return self


Condition = FrozenSet[Tuple[Variable, bool]]


class ConditionalDistribution(Distribution, ABC):
    pass


def instantiations(variables: Iterable[Variable], assignments: Optional[Dict[Variable, bool]] = None) -> Iterable[Dict[Variable, bool]]:
    assignments = assignments or {}
    return ({variable: value for variable, value in instance}
            for instance in product(*[
                ([(variable, assignments[variable])] if variable in assignments else [(variable, True), (variable, False)])
                for variable in variables]))


class TabledConditionalDistribution(ConditionalDistribution):
    shape: str = 'shape = square'

    def __init__(self, probability_table: Dict[Condition, float]):
        conditions: Set[Variable] = set()
        for condition, true_probability in probability_table.items():
            assert 0.0 <= true_probability <= 1.0
            for variable, value in condition:
                conditions.add(variable)
        assert 2 ** len(conditions) == len(probability_table)

        super().__init__(conditions=conditions)
        self.probability_table: Dict[Condition, float] = probability_table.copy()

    def prune_condition(self, subject: Variable, variable: Variable, value: bool) -> Distribution:
        if variable not in self.conditions:
            raise ValueError()
        elif 1 < len(self.conditions):
            self.probability_table: Dict[Condition, float] = {
                frozenset((parent, cond_value) for parent, cond_value in condition if parent != variable): true_probability
                for condition, true_probability in self.probability_table.items() if (variable, value) in condition}
            self.conditions.remove(variable)
            return self
        else:
            return PriorDistribution(true_probability=self.query(subject=subject, assignments={variable: value, subject: True}))

    def query(self, subject: Variable, assignments: Dict[Variable, bool]) -> float:
        true_probability = self.probability_table[frozenset((var, value) for var, value in assignments.items() if var in self.conditions)]
        return true_probability if assignments[subject] else 1 - true_probability

    def localize(self, subject: Variable, assignments: Dict[Variable, bool]) -> PriorDistribution:
        probability = 0

        subject_value = assignments.get(subject, True)
        for instance in instantiations(self.conditions | {subject}, {**assignments, subject: subject_value}):
            probability += self.query(subject=subject, assignments=instance) * prod(variable.query(instance) for variable in self.conditions)

        return PriorDistribution(true_probability=probability if subject_value else 1 - probability)

    def local_belief(self, subject: Variable, assignments: Dict[Variable, bool], parents: Dict[Variable, float]) -> float:
        fixture = {subject: assignments[subject]}

        local_conditions = set(parents.keys())
        if len(local_conditions) < len(self.conditions):
            u_i = (self.conditions - local_conditions).pop()
            fixture[u_i] = assignments[u_i]

        return sum(self.query(subject=subject, assignments={**instance, **fixture}) *
                   prod((parents[u_k] if value else 1 - parents[u_k]) for u_k, value in instance.items())
                   for instance in instantiations(local_conditions, assignments))

    def to_cpd(self, subject: Variable) -> 'TabledConditionalDistribution':
        return self


class NoisyDistribution(ConditionalDistribution, ABC):
    def to_cpd(self, subject: Variable) -> TabledConditionalDistribution:
        return TabledConditionalDistribution(
            probability_table={
                frozenset(condition): self.query(subject, {var: value for var, value in condition + ((subject, True),)})
                for condition in product(*[[(var, True), (var, False)] for var in self.conditions])})


class NoisyOrDistribution(NoisyDistribution):
    shape: str = 'shape = circle, style = dashed'

    def __init__(self, probability_table: Dict[Variable, Tuple[float, float]], leaky_bias: float = 0):
        conditions: Set[Variable] = set()
        for variable, (true_probability, false_probability) in probability_table.items():
            assert 0.0 <= true_probability <= 1.0
            assert 0.0 <= false_probability <= 1.0
            conditions.add(variable)
        assert 0.0 <= leaky_bias <= 1.0

        super().__init__(conditions=conditions)
        self.probability_table: Dict[Variable, Tuple[float, float]] = probability_table.copy()
        self.leaky_bias: float = leaky_bias
        self.pruning_bias: float = 1

    def prune_condition(self, subject: Variable, variable: Variable, value: bool) -> Distribution:
        if variable not in self.conditions:
            raise ValueError()
        elif 1 < len(self.conditions):
            true_probability, false_probability = self.probability_table[variable]
            del self.probability_table[variable]
            self.pruning_bias *= 1 - (true_probability if value else false_probability)
            self.conditions.remove(variable)
            return self
        else:
            return PriorDistribution(true_probability=self.query(subject=subject, assignments={variable: value, subject: True}))

    def query(self, subject: Variable, assignments: Dict[Variable, bool]) -> float:
        true_probability = 1 - (1 - self.leaky_bias) * self.pruning_bias * \
                           prod(1 - self.probability_table[var][0 if assignments[var] else 1] for var in self.conditions)
        return true_probability if assignments[subject] else 1 - true_probability

    def localize(self, subject: Variable, assignments: Dict[Variable, bool]) -> PriorDistribution:
        u: Dict[Variable, float] = {var: true_probability for var, (true_probability, false_probability) in self.probability_table.items()}
        v: Dict[Variable, float] = {var: false_probability for var, (true_probability, false_probability) in self.probability_table.items()}
        p: Dict[Variable, float] = {var: var.query({**assignments, var: True}) for var in self.conditions}
        return PriorDistribution(
            true_probability=1 - (1 - self.leaky_bias) * self.pruning_bias * prod((1 - v[var] + (v[var] - u[var]) * p[var]) for var in self.conditions))

    def local_belief(self, subject: Variable, assignments: Dict[Variable, bool], parents: Dict[Variable, float]) -> float:
        u: Dict[Variable, float] = {var: true_probability for var, (true_probability, false_probability) in self.probability_table.items()}
        v: Dict[Variable, float] = {var: false_probability for var, (true_probability, false_probability) in self.probability_table.items()}
        p: Dict[Variable, float] = parents
        true_probability = 1 - (1 - self.leaky_bias) * self.pruning_bias * prod((1 - v[var] + (v[var] - u[var]) * p[var]) for var in self.conditions)
        return true_probability if assignments[subject] else 1 - true_probability


class NoisyAndDistribution(NoisyDistribution):
    shape: str = 'shape = Mcircle'

    def __init__(self, probability_table: Dict[Variable, Tuple[float, float]], leaky_bias: float = 0):
        conditions: Set[Variable] = set()
        for variable, (true_probability, false_probability) in probability_table.items():
            assert 0.0 <= true_probability <= 1.0
            assert 0.0 <= false_probability <= 1.0
            conditions.add(variable)
        assert 0.0 <= leaky_bias <= 1.0

        super().__init__(conditions=conditions)
        self.probability_table: Dict[Variable, Tuple[float, float]] = probability_table.copy()
        self.leaky_bias: float = leaky_bias
        self.pruning_bias: float = 1

    def prune_condition(self, subject: Variable, variable: Variable, value: bool):
        if variable not in self.conditions:
            raise ValueError()
        elif 1 < len(self.conditions):
            true_probability, false_probability = self.probability_table[variable]
            del self.probability_table[variable]
            self.pruning_bias *= true_probability if value else false_probability
            self.conditions.remove(variable)
            return self
        else:
            return PriorDistribution(true_probability=self.query(subject=subject, assignments={variable: value, subject: True}))

    def query(self, subject: Variable, assignments: Dict[Variable, bool]) -> float:
        true_probability = (1 - self.leaky_bias) * self.pruning_bias * \
                           prod(self.probability_table[var][0 if assignments[var] else 1] for var in self.conditions)
        return true_probability if assignments[subject] else 1 - true_probability

    def localize(self, subject: Variable, assignments: Dict[Variable, bool]) -> PriorDistribution:
        u: Dict[Variable, float] = {var: true_probability for var, (true_probability, false_probability) in self.probability_table.items()}
        v: Dict[Variable, float] = {var: false_probability for var, (true_probability, false_probability) in self.probability_table.items()}
        p: Dict[Variable, float] = {var: var.query({**assignments, var: True}) for var in self.conditions}
        return PriorDistribution(
            true_probability=(1 - self.leaky_bias) * self.pruning_bias * prod((v[var] + (u[var] - v[var]) * p[var]) for var in self.conditions))

    def local_belief(self, subject: Variable, assignments: Dict[Variable, bool], parents: Dict[Variable, float]) -> float:
        u: Dict[Variable, float] = {var: true_probability for var, (true_probability, false_probability) in self.probability_table.items()}
        v: Dict[Variable, float] = {var: false_probability for var, (true_probability, false_probability) in self.probability_table.items()}
        p: Dict[Variable, float] = parents
        true_probability = (1 - self.leaky_bias) * self.pruning_bias * prod((v[var] + (u[var] - v[var]) * p[var]) for var in self.conditions)
        return true_probability if assignments[subject] else 1 - true_probability


class BayesianNet:
    def __init__(self, variables: Optional[Set[Variable]] = None, verify: bool = False):
        self.variables: Set[Variable] = set(variables or [])
        if variables and verify:
            self.verify()
        self.node_pruning = True
        self.edge_pruning = True
        self.subtree_pruning = True
        self.pie_msg: Dict[Tuple[Variable, Variable, bool], float] = defaultdict(neutral)
        self.lambda_msg: Dict[Tuple[Variable, Variable, bool], float] = defaultdict(neutral)

    def add(self, variable: Variable, verify: bool = True):
        self.variables.add(variable)
        if verify:
            self.verify()
        return self

    def verify(self):
        assert reduce(lambda a, b: a | b, (var.parents for var in self.variables)).issubset(self.variables)

    def clone(self) -> 'BayesianNet':
        return pickle.loads(pickle.dumps(self))

    def to_cpd(self):
        for variable in self.variables:
            variable.to_cpd()

    def _node_pruning(self, assignments: Dict[Variable, bool]):
        children = children_map(self.variables)
        observed_variables = set(assignments.keys())
        while True:
            for variable in self.variables.copy():
                if not children[variable] and variable not in observed_variables:
                    self.variables.remove(variable)
                    break
            else:
                break

    def _edge_pruning(self, assignments: Dict[Variable, bool]):
        children = children_map(self.variables)
        for variable, value in assignments.items():
            for condition_fixed_variable in children[variable]:
                condition_fixed_variable.prune_parent(variable, value)

    def _subtree_pruning(self, assignments: Dict[Variable, bool]):
        children = children_map(self.variables)
        leaves: Set[Variable] = {variable
                                 for variable, _children in children.items()
                                 if not variable.parents and len(_children) == 1}

        if not leaves:
            return

        leaf_children: Set[Variable] = reduce(lambda a, b: a | b, (children[leaf] for leaf in leaves))

        subtree_pruned = True
        while subtree_pruned:
            subtree_pruned = False
            for leaf_child in leaf_children:
                if leaf_child.parents.issubset(leaves):
                    leaves -= leaf_child.parents
                    leaf_children = leaf_children - {leaf_child} | children[leaf_child]
                    self.variables -= leaf_child.parents
                    leaf_child.localize(assignments)
                    subtree_pruned = True

    def query(self, assignments: Dict[Variable, bool]) -> float:
        assert set(assignments.keys()) & self.variables

        # pruning
        if self.node_pruning:
            self._node_pruning(assignments)

        if self.edge_pruning:
            self._edge_pruning(assignments)

        if self.subtree_pruning:
            self._subtree_pruning(assignments)

        # marginalization
        probability = 0
        for instance in instantiations(self.variables, assignments):
            instance_probability = 1
            for variable in self.variables:
                instance_probability *= variable.query(instance)
            probability += instance_probability

        return probability

    def query_by_name(self, **kwargs: bool) -> float:
        bn = self.clone()
        dictionary = {var.name: var for var in bn.variables}
        return bn.query({dictionary[name]: value for name, value in kwargs.items()})

    def query_all(self) -> Dict[Variable, bool]:
        return {subject: self.query({subject: True}) for subject in self.variables}

    def ibp(self, assignments: Dict[Variable, bool]):
        children = children_map(self.variables)

        # belief propagation
        pie_msg: Dict[Tuple[Variable, Variable, bool], float] = defaultdict(neutral)
        lambda_msg: Dict[Tuple[Variable, Variable, bool], float] = defaultdict(neutral)

        message_change = 1
        while 0.0001 < message_change:
            _pie_msg: Dict[Tuple[Variable, Variable, bool], float] = defaultdict(neutral)
            _lambda_msg: Dict[Tuple[Variable, Variable, bool], float] = defaultdict(neutral)

            message_change = 0
            for x in self.variables:
                x_values = {assignments[x]} if x in assignments else {True, False}

                # lambda message update
                for u_i in x.parents:
                    true_case_msg = sum(prod(lambda_msg[y_j, x, x_value] for y_j in children[x]) *
                                        x.local_belief(assignments={**assignments, x: x_value, u_i: True},
                                                       parents={u_k: pie_msg[x, u_k, True] if u_k != u_i else 1 for u_k in x.parents})
                                        for x_value in x_values)

                    false_case_msg = sum(prod(lambda_msg[y_j, x, x_value] for y_j in children[x]) *
                                         x.local_belief(assignments={**assignments, x: x_value, u_i: False},
                                                        parents={u_k: pie_msg[x, u_k, True] if u_k != u_i else 0 for u_k in x.parents})
                                         for x_value in x_values)

                    _lambda_msg[x, u_i, True] = true_case_msg / (true_case_msg + false_case_msg)
                    _lambda_msg[x, u_i, False] = false_case_msg / (true_case_msg + false_case_msg)

                    message_change += abs(_lambda_msg[x, u_i, True] - lambda_msg[x, u_i, True])
                    message_change += abs(_lambda_msg[x, u_i, False] - lambda_msg[x, u_i, False])

                # pie message update
                for y_j in children[x]:
                    if x.parents:
                        true_case_msg = prod(lambda_msg[y_k, x, True] for y_k in children[x] if y_k != y_j) * \
                                        x.local_belief(assignments={**assignments, x: True},
                                                       parents={u_i: pie_msg[x, u_i, True] for u_i in x.parents})

                        false_case_msg = prod(lambda_msg[y_k, x, False] for y_k in children[x] if y_k != y_j) * \
                                         x.local_belief(assignments={**assignments, x: False},
                                                        parents={u_i: pie_msg[x, u_i, True] for u_i in x.parents})
                    else:
                        true_case_msg = x.local_belief(assignments={x: True}, parents={}) if x not in assignments or assignments[x] else 0
                        false_case_msg = x.local_belief(assignments={x: False}, parents={}) if x not in assignments or not assignments[x] else 0

                    _pie_msg[y_j, x, True] = true_case_msg / (true_case_msg + false_case_msg)
                    _pie_msg[y_j, x, False] = false_case_msg / (true_case_msg + false_case_msg)

                    message_change += abs(_pie_msg[y_j, x, True] - pie_msg[y_j, x, True])
                    message_change += abs(_pie_msg[y_j, x, False] - pie_msg[y_j, x, False])

            pie_msg = _pie_msg
            lambda_msg = _lambda_msg

        self.pie_msg = pie_msg
        self.lambda_msg = lambda_msg

    def query_by_ibp(self, subject: Variable, evidence: Dict[Variable, bool]) -> float:
        self.ibp(evidence)

        children = children_map(self.variables)

        true_case = sum(subject.query({**instance, subject: True}) *
                        prod(self.pie_msg[subject, parent, instance[parent]] for parent in subject.parents) *
                        prod(self.lambda_msg[child, subject, True] for child in children[subject])
                        for instance in instantiations(subject.parents, evidence))

        false_case = sum(subject.query({**instance, subject: False}) *
                         prod(self.pie_msg[subject, parent, instance[parent]] for parent in subject.parents) *
                         prod(self.lambda_msg[child, subject, False] for child in children[subject])
                         for instance in instantiations(subject.parents, evidence))

        return true_case / (true_case + false_case)

    def query_all_by_ibp(self, evidence: Dict[Variable, bool]) -> Dict[Variable, float]:
        self.ibp(evidence)

        children = children_map(self.variables)

        reachabilites: Dict[Variable, float] = {}

        for subject in self.variables:
            true_case = sum(subject.query({**instance, subject: True}) *
                            prod(self.pie_msg[subject, parent, instance[parent]] for parent in subject.parents) *
                            prod(self.lambda_msg[child, subject, True] for child in children[subject])
                            for instance in instantiations(subject.parents, evidence))

            false_case = sum(subject.query({**instance, subject: False}) *
                             prod(self.pie_msg[subject, parent, instance[parent]] for parent in subject.parents) *
                             prod(self.lambda_msg[child, subject, False] for child in children[subject])
                             for instance in instantiations(subject.parents, evidence))

            reachabilites[subject] = true_case / (true_case + false_case)

        return reachabilites

    def query_one_by_ibp(self, subject_name: str) -> float:
        dictionary = {var.name: var for var in self.variables}
        return self.query_by_ibp(subject=dictionary[subject_name], evidence={})

    def to_dot(self, reachabilities: Optional[Dict[Variable, float]] = None) -> str:
        with open('bayesian_network_template.dot', 'r') as fp:
            template = fp.read()

        reachabilities = reachabilities or {}
        node_defs = [f'"{variable.name}" [{variable.distribution.shape}{f", label = {round(reachabilities[variable], 3)}" if variable in reachabilities else ""}];' for variable in self.variables]
        edge_defs = [f'"{parent.name}" -> "{child.name}";' for child in self.variables for parent in child.parents]
        return template.replace('%Nodes%', '\n  '.join(node_defs)).replace('%Edges%', '\n  '.join(edge_defs))


@contextmanager
def bayesian_network(dictionary: Dict[str, Any]) -> BayesianNet:

    keys_before = set(dictionary.keys()).copy()
    bn = BayesianNet()

    def commit(dictionary: Dict[str, Any]) -> BayesianNet:
        for new_key in set(dictionary.keys()) - keys_before:
            new_value = dictionary[new_key]
            if isinstance(new_value, Variable) and not new_value.name:
                new_value.name = new_key
                bn.add(new_value, verify=False)
        return bn

    yield commit

    bn.verify()


