from sklearn.datasets.samples_generator import make_blobs
from pandas import DataFrame
import matplotlib.pyplot as plt
import exercise2.Kernel as kernel
from exercise2.SVM import SVM
import numpy as np
import tools_Plot as t_pl

# installs:
#   scikit-learn
#   pandas
#   cvxopt

##part 1: choosing data
# training data:


X, t = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=0.9)
X = np.transpose(X)
mask = t <= 0
t[mask] = -1

d = DataFrame(dict(x=X[0, :], y=X[1, :], label=t))
figure, axis = plt.subplots()

grouped = d.groupby('label')
# color dictionary for plot
colors = {-1: 'red', 1: 'blue'}

# draw scatter plot
for i, group in grouped:
    color = colors[i]
    group.plot(ax=axis, kind='scatter', x='x', y='y', label=i, color=color)

#plt.show()


# part 2: SVM
svm = SVM()
[alpha, w0,positions] = svm.trainSVM(X, t, kernel.linearkernel)


t_pl.plot_SVM(alpha,w0,positions,X,t,X,t,kernel.linearkernel)
# res = svm.trainSVM(X,t)
