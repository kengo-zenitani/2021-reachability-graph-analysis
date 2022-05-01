# Overview

This package provides the source code files for the experiments
in "A scalable algorithm for network reachability analysis with cyclic attack graphs".
The programs are written in Python 3.7.0. We use cpython3.7 to perform the benchmarks.
The quick summary of each file is shown below.

*   **setup.py**: randomly generates the network models for the experiments. The generated files will be stored in `./network/`.
*   **test2_bn.py**, **test2_rg.py**: performs the benchmarking to compare the inference speed of LBP and reachability graph analysis. The result files will be stored in `./benchmark/test2-bn/` and `./benchmark/test2-rg/`, respectively.
*   **test3.py**: performs the benchmarking to verify the scalability of reachability graph analysis. The result files will be stored in `./benchmark/test3/`.
*   **test4.py**: performs the benchmarking to verify the performance of reachability graph analysis applied to cyclic attack graphs. The result files will be stored in `./benchmark/test4/`.
*   **utils_figure.py**: generates the figures.

Other files below provide the foundation for the programs above.

*   **common.py**: defines the common classes.
*   **network.py**: defines the classes which implement the network model.
*   **network_template.dot**: is used by the method `Network.to_dot` in **network.py**.
*   **bayesian_network.py**: implements the Bayesian inference based on LBP algorithm.
*   **bayesian_network_template.dot**: is used by the method `BayesianNet.to_dot` in **bayesian_network.py**.
*   **reachability_graph.py**: implements the reachability graph analysis algorithm.
*   **reachability_graph_template.dot**: is used by the method `ReachabilityGraph.to_dot` in **reachability_graph.py**.

# Setup

Python 3.7.0 environment with [matplotlib](https://matplotlib.org/) is required.
To generate the network models, just to type and enter the following.

```
$ python setup.py
```

This setup is needed before the test programs execution below.

# Note

*   You can preview the network structure by `Network.to_dot`.
The output is a .dot script for [graphviz](https://www.graphviz.org/).
Similarly, `BayesianNet.to_dot` and `ReachabilityGraph.to_dot` generate .dot scripts for a Bayesian attack graph and a reachability graph respectively.

*   In the paper, both a Bayesian attack graph and a reachability graph are supposed to be generated from an attack graph.
However, in this program, both graphs are generated directly from the abstract network model defined as an instance of Network.
The conversion process is implemented in `ReachabilityEvaluator.__init__` and `BayesianEvaluator.__init__`.
A Network instance is not an attack graph, which is equivalent to the composition of an attack graph *G*, and initial reachabilities *I*.
(In the program, initial reachabilities are coded as "rigidity". More precisely, `(1 - rigidity) / 100 == initial_reachability`.)

*   The central part of the implementation of reachability graph analysis is in `ReachabilityGraph.query`.
