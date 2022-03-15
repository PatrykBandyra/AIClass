"""Patryk Bandyra"""

import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt

"""
Assumption: no member of population is allowed to not meet cover condition in any moment
"""


def return_complete_graph(n_nodes):
    """
    :param n_nodes: number of nodes
    :return: nx.Graph
    """
    return nx.complete_graph(range(n_nodes))


def check_cover_condition(member, graph):
    """
    :param member: list of covered vertices (1 or 0)
    :param graph: nx.Graph
    :return: True if condition is met
    """
    for edge in graph.edges:
        if member[edge[0]] != 1 and member[edge[1]] != 1:
            return False
    return True


def initialize_population(pop_size, graph, p, debug=False):
    """
    Population is initialized with only those members that meet cover condition
    :param pop_size: number of members in population
    :param graph: nx.Graph
    :param p: probability of choosing 1 to insert to new_member
    :return population (list of lists)
    """
    if debug:
        print('Initializing starting population')
    population = []
    for i in range(pop_size):
        while True:
            new_member = []
            for _ in range(len(graph.nodes)):
                new_member.append(np.random.binomial(1, p))
            if check_cover_condition(new_member, graph) is True:
                population.append(new_member)
                if debug:
                    print(f'New member: {new_member}\tSum: {sum(new_member)}')
                break
    if debug:
        print('Population initialized')
    return population


def locate_min(list):
    """
    :param list: list
    :return: smallest value from list, indexes
    """
    smallest = min(list)
    return smallest, [index for index, element in enumerate(list)
                      if smallest == element]


def locate_max(list):
    """
    :param list: list
    :return: biggest value from list, indexes
    """
    biggest = max(list)
    return biggest, [index for index, element in enumerate(list)
                      if biggest == element]


def tournament_selection(population, k, debug=False):
    """
    :param population: list of members (list of lists)
    :param k: number of members taking part in each tournament
    :return: temporary population to be mutated
    """
    if debug:
        print('Tournament...')
    temp_population = []
    for _ in range(math.ceil(len(population)/k)):
        tournament = []
        # choose members to tournament
        for _ in range(k):
            tournament.append(population[np.random.randint(0, len(population))])
        # choose best
        smallest, winners_indexes = locate_min(tournament)
        winner_index = np.random.choice(winners_indexes)
        temp_population.append(tournament[winner_index])    # add the winner to return population
        del(population[winner_index])   # remove the winner from population
    if debug:
        print('End of tournament')
    return temp_population


def mutate(population, mut_p, max_tries, graph, debug=False):
    """
    If member of population is chosen to be mutated:
    1. Find random positive gene (1 - node covered) and make it negative
    2. If cover condition is not met - undo and continue trying till condition is met or
    number of tries equals maximum
    :param population: members that might be mutated
    :param mut_p: probability of mutation
    :param max_tries: max number of tries to mutate chosen member
    :param graph: nx.Graph
    :return:
    """
    if debug:
        print('Mutating...')
    for member in population:
        if np.random.uniform(0, 1) < mut_p:     # mutate
            # create look-up table of indexes where gene equals 1
            lookuptable = [i for i in range(len(member)) if member[i] == 1]
            j = 0
            while j < max_tries:
                # choose random positive gene
                index = np.random.choice(lookuptable)
                member[index] = 0
                if check_cover_condition(member, graph):
                    if debug:
                        print('Mutation succeeded')
                    break
                # condition not met
                member[index] = 1
                lookuptable.remove(index)
                j += 1
    if debug:
        print('End of mutation')
    return population


def algorithm(pop_size, graph, p, max_eval_n, k, mut_p, max_tries, debug=False):
    """
    :param pop_size: number of members in population
    :param graph: nx.Graph
    :param p: probability of choosing 1 to insert to new_member (during pop. init.)
    :param max_eval_n: maximal number of calls of evaluation function
    :param k: number of members taking part in each tournament
    :param mut_p: mutation probability
    :param max_tries: max number of tries to mutate chosen member
    :return: last generation (list of lists)
    """
    # population initialization
    population = initialize_population(pop_size, graph, p, debug)
    temp_population = []
    mutated_population = []
    t = 0   # number of calls of evaluation function
    while t < max_eval_n:
        # tournament selection
        temp_population = tournament_selection(population.copy(), k, debug=debug)
        mutated_population = mutate(temp_population, mut_p, max_tries, graph, debug=debug)
        # combine both mutated population and population from previous generation
        population_sum = population + mutated_population
        population.clear()
        # choose best members
        if debug:
            print('Succession...')
        lookuptable = [sum(member) for member in population_sum]    # lower = better
        while len(population) < pop_size:
            value, bests_indexes = locate_min(population_sum)
            population.append(population_sum.pop(np.random.choice(bests_indexes)))  # choose random from bests
            t += 1
        if debug:
            print('End of succession')
            print(f'List of numbers of covered nodes: {[sum(m) for m in population]}')
    return population


def test_complete_graph():
    graph = return_complete_graph(25)
    population = algorithm(100, graph, 0.8, 1000, 2, 0.5, 3)
    print('Complete graph')
    print(f'Last generation: {population}')
    print(f'List of numbers of covered nodes: {[sum(m) for m in population]}')
    print(f'Length: {len(population)}')
    print('Red - covered, Cyan - not covered')
    map_color = []
    for i in population[0]:
        if i == 1:
            map_color.append('red')
        else:
            map_color.append('cyan')
    nx.draw(graph, node_color=map_color, with_labels=True)
    plt.show()


def test_random_bipartite_graph():
    graph = nx.algorithms.bipartite.generators.random_graph(12, 13, 0.5)
    population = algorithm(100, graph, 0.8, 1000, 2, 0.5, 3)
    print('Bipartite graph')
    print(f'Last generation: {population}')
    print(f'List of numbers of covered nodes: {[sum(m) for m in population]}')
    print(f'Length: {len(population)}')
    print('Red - covered, Cyan - not covered')
    map_color = []
    for i in population[0]:
        if i == 1:
            map_color.append('red')
        else:
            map_color.append('cyan')
    nx.draw(graph, node_color=map_color, with_labels=True, label='Red: covered, Cyan: not covered')
    plt.show()


def test_random_graph():
    """
    Might be connected or disconnected
    """
    graph = nx.generators.random_graphs.gnm_random_graph(25, 33)
    population = algorithm(100, graph, 0.8, 1000, 2, 0.5, 3)
    print('Bipartite graph')
    print(f'Last generation: {population}')
    print(f'List of numbers of covered nodes: {[sum(m) for m in population]}')
    print(f'Length: {len(population)}')
    print('Red - covered, Cyan - not covered')
    map_color = []
    for i in population[0]:
        if i == 1:
            map_color.append('red')
        else:
            map_color.append('cyan')
    nx.draw(graph, node_color=map_color, with_labels=True, label='Red: covered, Cyan: not covered')
    plt.show()


def test_population_size_dependency():
    populations = [10, 50, 100, 1000, 2000]
    graph = nx.generators.random_graphs.gnm_random_graph(25, 33)
    results = []
    for population in populations:
        result = algorithm(population, graph, 0.8, 1000, 2, 0.5, 3)
        sums = [sum(m) for m in result]
        min_value = min(sums)
        max_value = max(sums)
        results.append((min_value, max_value))

    for xe, ye in zip(populations, results):
        plt.scatter([xe] * len(ye), ye)
    plt.xticks([10, 50, 100, 1000, 2000])
    plt.axes().set_xticklabels(populations)
    plt.show()


def test_mutation_prob_dependency():
    prob_mut = [0.2, 0.4, 0.6, 0.8]
    graph = nx.generators.random_graphs.gnm_random_graph(25, 33)
    results = []
    for p in prob_mut:
        result = algorithm(1000, graph, 0.8, 1000, 2, p, 3)
        sums = [sum(m) for m in result]
        min_value = min(sums)
        max_value = max(sums)
        results.append((min_value, max_value))
    for xe, ye in zip(prob_mut, results):
        plt.scatter([xe] * len(ye), ye)
    plt.xticks([0.2, 0.4, 0.6, 0.8])
    plt.axes().set_xticklabels(prob_mut)
    plt.show()


def test_k_param_dependency():  # k - number of members in each tournament
    ks = [2, 3, 4, 5]
    graph = nx.generators.random_graphs.gnm_random_graph(25, 33)
    results = []
    for k in ks:
        result = algorithm(1000, graph, 0.8, 1000, k, 0.5, 3)
        sums = [sum(m) for m in result]
        min_value = min(sums)
        max_value = max(sums)
        results.append((min_value, max_value))
    for xe, ye in zip(ks, results):
        plt.scatter([xe] * len(ye), ye)
    plt.xticks([2, 3, 4 ,5])
    plt.axes().set_xticklabels(ks)
    plt.show()


if __name__ == '__main__':
    # test_mutation_prob_dependency()
    # test_population_size_dependency()
    # test_k_param_dependency()
    test_random_graph()
    test_complete_graph()
    test_random_bipartite_graph()
