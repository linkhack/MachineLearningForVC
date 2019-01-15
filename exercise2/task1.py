from sklearn.datasets.samples_generator import make_blobs
from pandas import DataFrame
import matplotlib.pyplot as plt
import exercise2.Kernel as kernel
import numpy as np
import exercise2.tools_Plot as plot
from exercise2.SVM import SVM

# installs:
#   scikit-learn
#   pandas
#   cvxopt

##part 1: choosing data
# training data:


X, t = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=0.9)
# assign either 1 or -1 as label!
mask = t <= 0
t[mask] = -1

d = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=t))
figure, axis = plt.subplots()

# group by label obviously
grouped = d.groupby('label')
# color dictionary for plot
colors = {-1: 'red', 1: 'blue'}

# part 2: SVM with hard margin
svm = SVM()
svm.setSigma(1.0)

X = np.transpose(X)
# svm with linear kernel
#[alpha, w0,sv_index] = svm.trainSVM(X, t)


# svm with linear kernel and slack variable
#[alpha, w0,sv_index] = svm.trainSVM(X, t, kernel.linearkernel, 0.1)

# svm with rbf kernel
# [alpha, w0, sv_index] = svm.trainSVM(X, t, kernel.rbfkernel )

# svm with rbf kernel and slack
[alpha, w0, sv_index] = svm.trainSVM(X, t, kernel.rbfkernel, 1.5)


# draw data plot
for i, group in grouped:
    color = colors[i]
    group.plot(ax=axis, kind='scatter', x='x', y='y', label=i, color=color)

X = np.transpose(X)
#draw margin
plot.plot(X, t, w0, alpha, sv_index, svm)
# tadaaaaa
