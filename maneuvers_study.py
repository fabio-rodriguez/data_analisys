import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import *


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


df = pd.read_excel('data/outputs_concat2.xlsx', index_col=0)

maneuvers=np.array(df['out_action'])
id_in_seq=np.array(df['id_in_seq'])


columns = ["id_in_seq", "out_action"]

dd = {id:{} for id in id_in_seq}


for index, i in enumerate(id_in_seq):

    try: 
        dd[i][maneuvers[index]] += 1
    except:
        dd[i][maneuvers[index]] = 1

print(dd)

maneuvers = {}
for key in dd.keys():
    dman = dd[key]
    if key == 0:
        continue
    for m, times in dman.items():
        try:
            maneuvers[m] += times
        except:
            maneuvers[m] = times

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
mans, times = zip(*maneuvers.items())
ax.bar(mans,times)
ax.set_xticklabels(mans)
plt.show()


transitions = transitions_preperching()
print(len(transitions))

for key in dd.keys():
    if key == 0:
        continue

