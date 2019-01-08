from sklearn.datasets.samples_generator import make_blobs
from pandas import DataFrame
import matplotlib.pyplot as plt
import exercise2.Kernel as kernel
import numpy as np
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

# part 2: SVM
svm = SVM()
# [sv, sv_X, sv_T, w0] = svm.trainSVM(X, t, kernel.linearkernel)
[alpha, w0] = svm.trainSVM(X, t, kernel.linearkernel)

# res = svm.trainSVM(X,t)
# draw scatter plot
for i, group in grouped:
    color = colors[i]
    group.plot(ax=axis, kind='scatter', x='x', y='y', label=i, color=color)

# Support vectors have non zero lagrange multipliers
sv_index = alpha > 1e-5  # some small threshold a little bit greater than 0, [> 0  was too crowded]

# position index of support vectors in alpha array
ind = np.arange(len(alpha))[sv_index]

# get support vectors and corresponding x and label values
sv = alpha[sv_index]
sv_X = X[sv_index]
sv_T = t[sv_index]



# draw margin:
# get min and max values of training dimensions first
minX = min(X[:, 0])
maxX = max(X[:, 0])
minY = min(X[:, 1])
maxY = max(X[:, 1])

#create meshgrid for current space
X1, X2 = np.meshgrid(np.linspace(minX, maxX, 50), np.linspace(minY, maxY, 50))
X_new = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
# calculate discriminante for all points in space
Z = svm.discriminant(alpha, w0, X, t, X_new).reshape(X1.shape)

plt.contour(X1, X2, Z, [0.0], colors='grey')
plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linestyles='dashed')
plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linestyles='dashed')


# show support vectors.
plt.scatter(sv_X[:, 0], sv_X[:, 1], s=50, c="g")
plt.show()
