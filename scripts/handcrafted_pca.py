import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    'var1': [1,3,1,2,2,1],
    'var2':[2,4,3,4,3,4],
    'target': [0,1,0,1,1,1]
})

X = df.drop('target', axis=1)
y = df['target']

mean_vec = np.mean(X, axis=0)
print(mean_vec)

M = X - mean_vec
print(M)

C = M.T.dot(M) / X.shape[0]-1
print(C)

eig_vals, eig_vecs = np.linalg.eig(C)

print('Eigenvalues \n%s' %eig_vals)
print('\nEigenvectors \n%s' %eig_vecs)