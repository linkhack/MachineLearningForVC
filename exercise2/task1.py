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


X, t = make_blobs(n_samples=200, centers=[[-2, -2], [2, 2]], n_features=2, cluster_std=0.9, random_state=6)
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
svm.setSigma(2.4)

# svm with linear kernel
#[alpha, w0, sv_index] = svm.trainSVM(X, t)

# svm with linear kernel and slack variable
[alpha, w0, sv_index] = svm.trainSVM(X, t, kernel.linearkernel, 1)

# svm with rbf kernel
#[alpha, w0, sv_index] = svm.trainSVM(X, t, kernel.rbfkernel,100.1)

# svm with rbf kernel and slack
# [alpha, w0, sv_index] = svm.trainSVM(X, t, kernel.rbfkernel,100)


# draw data plot
for i, group in grouped:
    color = colors[i]
    group.plot(ax=axis, kind='scatter', x='x', y='y', label=i, color=color)

# draw margin
plot.plot(X, t, w0, alpha, sv_index, svm)

# test SVM on new Dataset
# (same centers, same std_dev)->same distribution of data, other seed

X_test, t_test = make_blobs(n_samples=400, centers=[[-2, -2], [2, 2]], n_features=2, cluster_std=0.9, random_state=12)

mask = t_test <= 0
t_test[mask] = -1

predicted_labels_test = np.sign(svm.discriminant(alpha, w0, sv_index, X, t, X_test))

# Error rate
error_rate = plot.error_rate(predicted_labels_test, t_test)
print(f"Error rate is {error_rate}")
print(f"Wrongly classified {np.sum(t_test!=predicted_labels_test)} of {np.size(t_test)}")

# plot new data
d = DataFrame(dict(x=X_test[:, 0], y=X_test[:, 1], label=t_test))
figure, axis = plt.subplots()

# group by label obviously
grouped = d.groupby('label')

# draw data plot
for i, group in grouped:
    color = colors[i]
    group.plot(ax=axis, kind='scatter', x='x', y='y', label=i, color=color)

# draw decision boundry
plot.plot(X, t, w0, alpha, sv_index, svm)
plt.show()
# tadaaaaa
