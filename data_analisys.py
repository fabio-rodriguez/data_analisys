import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.decomposition import PCA


df = pd.read_excel('outputs_concat2.xlsx', index_col=0)

states_str=df['current_state']
states = np.array([eval(state) for state in states_str])

columns = ["U", "W", "Omega", 'Theta', 'X', 'Z']
df = pd.DataFrame(data=states, columns=columns)

#corrMatrix = df.corr()

#sn.heatmap(corrMatrix, annot=True)
#plt.show()

pca = PCA(n_components=6)
pca.fit(df)

print("explained ratio:", pca.explained_variance_ratio_) 
print("singular values:", pca.singular_values_) 
