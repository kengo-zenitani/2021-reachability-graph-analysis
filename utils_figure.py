from benchmark import *


def Figure_5_3():
    import matplotlib.pyplot as plt

    left = []
    height = []
    yerr = []
    for problem_class in ['nw1', 'nw2', 'nw3', 'nw4', 'nw5']:
        left.append(problem_class)
        performance = check_processing_times('test2-bn', problem_class)
        height.append(performance[0])
        yerr.append(performance[1])

    plt.subplots(figsize=(7, 5))
    plt.grid()
    plt.xlabel('network class')
    plt.ylabel('processing time (sec)')
    plt.bar(left, height, yerr=yerr, error_kw=dict(lw=1, capthick=1, capsize=20), color="#D0D0D0")
    plt.savefig('Figure-5.3.pdf')


def Figure_5_4():
    import matplotlib.pyplot as plt

    left = []
    height = []
    yerr = []
    for problem_class in ['nw1', 'nw2', 'nw3', 'nw4', 'nw5']:
        left.append(problem_class)
        performance = check_processing_times('test2-rg', problem_class)
        height.append(performance[0])
        yerr.append(performance[1])

    plt.subplots(figsize=(7, 5))
    plt.grid()
    plt.xlabel('network class')
    plt.ylabel('processing time (sec)')
    plt.bar(left, height, yerr=yerr, error_kw=dict(lw=1, capthick=1, capsize=20), color="#D0D0D0")
    plt.savefig('Figure-5.4.pdf')


def Figure_5_5():

    def compare_processing_times():
        results = {}
        for improved_algorithm_name, base_algorithm_name in [('test2-rg', 'test2-bn')]:
            for problem_class in ['nw1', 'nw2', 'nw3', 'nw4', 'nw5']:
                results_in_a_class = []
                for index in range(0, 50):
                    improved_result = Benchmark.load(algorithm_tag=improved_algorithm_name, problem_tag=f"{problem_class}-{index + 1}")
                    base_result = Benchmark.load(algorithm_tag=base_algorithm_name, problem_tag=f"{problem_class}-{index + 1}")
                    results_in_a_class.append(base_result.processing_time / improved_result.processing_time)
                results[problem_class] = results_in_a_class
        return results

    import matplotlib.pyplot as plt

    results = compare_processing_times()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot((results['nw1'], results['nw2'], results['nw3'], results['nw4'], results['nw5']), medianprops=dict(color="black"))
    ax.set_xticklabels(['nw1', 'nw2', 'nw3', 'nw4', 'mw5'])

    plt.xlabel('network class')
    plt.ylabel('processing time ratio (BAG/RG)')
    plt.grid()

    plt.savefig('Figure-5.5.pdf')


def Figure_5_6():
    import matplotlib.pyplot as plt

    left = []
    height = []
    yerr = []
    for problem_class in ['nw6', 'nw7', 'nw8', 'nw9', 'nw10']:
        left.append(problem_class)
        performance = check_processing_times('test3', problem_class)
        height.append(performance[0])
        yerr.append(performance[1])

    plt.subplots(figsize=(7, 5))
    plt.grid()
    plt.xlabel('network class')
    plt.ylabel('processing time (sec)')
    plt.bar(left, height, yerr=yerr, error_kw=dict(lw=1, capthick=1, capsize=20), color="#D0D0D0")
    plt.savefig('Figure-5.6.pdf')


def Figure_5_8():

    problem_classes = [f'nw10-h{h * 8}' for h in range(8)]

    def compare_processing_times():
        results = {}
        for problem_class in problem_classes:
            results_in_a_class = []
            for index in range(0, 50):
                result = Benchmark.load(algorithm_tag='test4', problem_tag=f"{problem_class}-{index + 1}")
                results_in_a_class.append(result.processing_time)
            results[problem_class] = results_in_a_class
        return results

    import matplotlib.pyplot as plt

    results = compare_processing_times()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(tuple(results[problem_class] for problem_class in problem_classes), medianprops=dict(color="black"))
    ax.set_xticklabels(['no hubs'] + [str(h * 8) for h in range(1, 8)])

    plt.xlabel('number of hubs in the gateway layer')
    plt.ylabel('processing time (sec)')
    plt.grid()

    plt.savefig('Figure-5.8.pdf')


def Figure_6_1():

    problem_classes = [('test2-rg', f'nw{i}') for i in range(1, 6)] + [('test3', f'nw{i}') for i in range(6, 11)]

    def compare_processing_times():
        results = {}
        for algorithm_tag, problem_class in problem_classes:
            results_in_a_class = []
            for index in range(0, 50):
                result = Benchmark.load(algorithm_tag=algorithm_tag, problem_tag=f"{problem_class}-{index + 1}")
                results_in_a_class.append(result.update_count / len(result.reachabilities))
            results[problem_class] = results_in_a_class
        return results

    import matplotlib.pyplot as plt

    results = compare_processing_times()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.boxplot(tuple(results[problem_class] for _, problem_class in problem_classes), medianprops=dict(color="black"))
    ax.set_xticklabels(problem_class for _, problem_class in problem_classes)

    plt.xlabel('update count per node')
    plt.grid()

    plt.savefig('Figure-6.1.pdf')


def Figure_6_2():

    problem_classes = [('test2-rg', f'nw{i}') for i in range(1, 6)] + [('test3', f'nw{i}') for i in range(6, 11)]

    def compare_processing_times():
        results = {}
        for algorithm_tag, problem_class in problem_classes:
            results_in_a_class = []
            for index in range(0, 50):
                result = Benchmark.load(algorithm_tag=algorithm_tag, problem_tag=f"{problem_class}-{index + 1}")
                results_in_a_class.append(result.processing_time / len(result.reachabilities))
            results[problem_class] = results_in_a_class
        return results

    import matplotlib.pyplot as plt

    results = compare_processing_times()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.boxplot(tuple(results[problem_class] for _, problem_class in problem_classes), medianprops=dict(color="black"))
    ax.set_xticklabels(problem_class for _, problem_class in problem_classes)

    plt.xlabel('processing time per node (msec)')
    plt.grid()

    plt.savefig('Figure-6.2.pdf')


def Figure_6_3():

    problem_classes = [('test2-rg', f'nw{i}') for i in range(1, 6)] + [('test3', f'nw{i}') for i in range(6, 11)]

    def compare_processing_times():
        results = {}
        for algorithm_tag, problem_class in problem_classes:
            results_in_a_class = []
            for index in range(0, 50):
                result = Benchmark.load(algorithm_tag=algorithm_tag, problem_tag=f"{problem_class}-{index + 1}")
                results_in_a_class.append(result.select_count / len(result.reachabilities))
            results[problem_class] = results_in_a_class
        return results

    import matplotlib.pyplot as plt

    results = compare_processing_times()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.boxplot(tuple(results[problem_class] for _, problem_class in problem_classes), medianprops=dict(color="black"))
    ax.set_xticklabels(problem_class for _, problem_class in problem_classes)

    plt.xlabel('evaluation count per node')
    plt.grid()

    plt.savefig('Figure-6.3.pdf')


def Figure_6_4():

    problem_classes = [f'nw10-h{h * 8}' for h in range(8)]

    def compare_processing_times():
        results = {}
        for problem_class in problem_classes:
            results_in_a_class = []
            for index in range(0, 50):
                result = Benchmark.load(algorithm_tag='test4', problem_tag=f"{problem_class}-{index + 1}")
                results_in_a_class.append(result.processing_time / len(result.reachabilities))
            results[problem_class] = results_in_a_class
        return results

    import matplotlib.pyplot as plt

    results = compare_processing_times()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(tuple(results[problem_class] for problem_class in problem_classes), medianprops=dict(color="black"))
    ax.set_xticklabels(['no hubs'] + [str(h * 8) for h in range(1, 8)])

    plt.xlabel('number of hubs in the gateway layer')
    plt.ylabel('processing time per node (msec)')
    plt.grid()

    plt.savefig('Figure-6.4.pdf')


def Figure_6_5():

    problem_classes = [f'nw10-h{h * 8}' for h in range(8)]

    def compare_processing_times():
        results = {}
        for problem_class in problem_classes:
            results_in_a_class = []
            for index in range(0, 50):
                result = Benchmark.load(algorithm_tag='test4', problem_tag=f"{problem_class}-{index + 1}")
                results_in_a_class.append(result.update_count / len(result.reachabilities))
            results[problem_class] = results_in_a_class
        return results

    import matplotlib.pyplot as plt

    results = compare_processing_times()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(tuple(results[problem_class] for problem_class in problem_classes), medianprops=dict(color="black"))
    ax.set_xticklabels(['no hubs'] + [str(h * 8) for h in range(1, 8)])

    plt.xlabel('number of hubs in the gateway layer')
    plt.ylabel('update count per node')
    plt.grid()

    plt.savefig('Figure-6.5.pdf')


def Figure_6_6():

    problem_classes = [f'nw10-h{h * 8}' for h in range(8)]

    def compare_processing_times():
        results = {}
        for problem_class in problem_classes:
            results_in_a_class = []
            for index in range(0, 50):
                result = Benchmark.load(algorithm_tag='test4', problem_tag=f"{problem_class}-{index + 1}")
                results_in_a_class.append(result.select_count / len(result.reachabilities))
            results[problem_class] = results_in_a_class
        return results

    import matplotlib.pyplot as plt

    results = compare_processing_times()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(tuple(results[problem_class] for problem_class in problem_classes), medianprops=dict(color="black"))
    ax.set_xticklabels(['no hubs'] + [str(h * 8) for h in range(1, 8)])

    plt.xlabel('number of hubs in the gateway layer')
    plt.ylabel('evaluation count per node')
    plt.grid()

    plt.savefig('Figure-6.6.pdf')


"""
Figure_5_3()
Figure_5_4()
Figure_5_5()
Figure_5_6()
Figure_5_8()
"""
#Figure_6_1()
#Figure_6_2()
#Figure_6_3()
#Figure_6_4()
Figure_6_5()
#Figure_6_6()
