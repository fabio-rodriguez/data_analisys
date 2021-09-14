import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


df = pd.read_excel('outputs_concat2.xlsx', index_col=0)

maneuvers=np.array(df['out_action'])
id_in_seq=np.array(df['id_in_seq'])


columns = ["id_in_seq", "out_action"]

dd = {id:{} for id in id_in_seq}

print(dd)

for index, i in enumerate(id_in_seq):

    try: 
        dd[i][maneuvers[index]] += 1
    except:
        dd[i][maneuvers[index]] = 1

print(dd)


# 
# states = np.array([eval(state) for state in states_str])

# 
# corrMatrix = df.corr()

# sn.heatmap(corrMatrix, annot=True)
# plt.show()

# pca = PCA(n_components=5)
# pca.fit(df)

# print(pca.components_) 
