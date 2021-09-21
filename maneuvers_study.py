import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import *

from matplotlib.pyplot import figure

def transitions_preperching():

    feasible_actions = [(round(radians(-a), 5), f) for a in range(5) for f in range(5)]

    transitions = {}
    for a in range(5):
        for f in range(5):
            values = [round(radians(-a+i), 5) for i in [-1, 0, 1]] + [f+i for i in [-1, 0, 1]]
            next = set(itertools.combinations(values, 2))
            action = (round(radians(-a), 5), f)
            transitions[action] = [m for m in next if m in feasible_actions]

    return transitions


def column_from_datasets(datasets, column='action_codes'):

    column_list = []
    for path in datasets:
        pdi = pd.read_csv(path)
        column_list += list(np.transpose(pdi[column])) 

    return column_list


def calc_action_frequencies(actions_frequency_from_datasets):
    
    actions_frequency_from_datasets.sort()

    actions, freq = [actions_frequency_from_datasets.pop(0)], [1]
    for action in actions_frequency_from_datasets[1:]:
        if action == actions[-1]:
            freq[-1] += 1
        else:
            actions.append(action)
            freq.append(1)

    return actions, freq


def draw_frequencies(N, actions, frequencies, resultpath):

    actions.sort()
    Xs = list(range(N))
    Ys = []
    zeros = []
    marker = 0
    for i in Xs:
        if marker<len(actions) and i==actions[marker]:
            Ys.append(frequencies[marker])
            zeros.append(0)
            marker += 1
        else:
            Ys.append(0)
            zeros.append(-1000)
            
    plt.bar(Xs, Ys)
    plt.bar(Xs, zeros)
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.bar(Xs,Ys)
    # ax.set_xticklabels(mans)
    plt.xticks(Xs)
    plt.savefig(resultpath)
    plt.show()
    plt.close()


def draw_frequencies_prob(N, actions, frequencies, resultpath):

    s = sum(frequencies)
    print(s, frequencies)

    fs = [100*f/s for f in frequencies]
    
    print(sum(fs), fs)

    figure(figsize=(8, 6), dpi=80)

    plt.bar(actions, fs)
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.bar(Xs,Ys)
    # ax.set_xticklabels(mans)
    plt.xticks(actions)
    plt.ylabel("%")
    plt.xlabel("ID_maneuvers")
    plt.ylim([0, 25])
    plt.savefig(resultpath, bbox_inches='tight')
    plt.show()
    plt.close()


def exp1(name="default"):
    
    transitions = transitions_preperching()

    datasets = ['data/landing_test_mlp_format.csv', 'data/landing_train_mlp_format.csv']
    actions = column_from_datasets(datasets)
    
    actions, frequencies = calc_action_frequencies(actions)

    for i, f in enumerate(frequencies[:]):
        if not f:
            frequencies.pop(i)
            actions.pop(i)
    
    draw_frequencies_prob(len(transitions), actions, frequencies, name)


def exp2(N, name="default"):
    
    transitions = transitions_preperching()

    datasets = ['data/landing_test_mlp_format.csv', 'data/landing_train_mlp_format.csv']
    actions = column_from_datasets(datasets)
    ids = column_from_datasets(datasets, 'id_trajectory')
    
    actions, frequencies = calc_first_n_action_frequencies(N, actions, ids)
    
    draw_frequencies(len(transitions), actions, frequencies, name)


def exp3(N, name):
    
    transitions = transitions_preperching()

    datasets = ['data/landing_test_mlp_format.csv', 'data/landing_train_mlp_format.csv']
    actions = column_from_datasets(datasets)
    ids = column_from_datasets(datasets, 'id_trajectory')
    
    actions, frequencies = calc_first_n_action_frequencies(N, actions, ids, reverse=True)
    
    draw_frequencies(len(transitions), actions, frequencies, name)



def calc_first_n_action_frequencies(N, actions, ids, reverse=False):

    if reverse:
        actions.reverse()
        ids.reverse()

    marker = 1
    first_actions = [actions[0]]
    for index, id in enumerate(ids[1:]):
        if id == ids[index-1]:
            if marker<N:
                marker+=1
                first_actions.append(actions[index])
        else:
            marker = 1
            first_actions.append(actions[index])
            
    return calc_action_frequencies(first_actions)
    

if __name__ == '__main__':

    exp1()

    # exp2(10)
    
    # for i in range(10):
    #     exp3(i+1, f'last_{i+1}')
    