import matplotlib.pyplot as plt
import matplotlib.image as pltImg
import numpy as np
import exercise2.Kernel as kernel

def plot_SVM(alpha,w0,positions,X,t,Xnew,tnew,kernel=kernel.linearkernel,sigma=-1):
    """ ploting of the decision boundaries and the margin for the SVM trained on (X,t) 
    assuming that X represent vector written by colums"""
    nFeatures, nSamples = X.shape
    min_x = np.min(Xnew[0,:])
    max_x = np.max(Xnew[0,:])

    min_y = np.min(Xnew[1,:])
    max_y = np.max(Xnew[1,:])
    
    x_axis = np.linspace(min_x,max_x,200)
    y_axis = np.linspace(min_y,max_y,200)
    
    X_grid, Y_grid = np.meshgrid(x_axis,y_axis)
    
    result = np.zeros([200,200])
    for i in range(0,200):
        for j in range(0,200):
            vector = np.transpose(np.array([[X_grid[i,j]],[Y_grid[i,j]]]))
            value = w0
            for k in range(nSamples):
                
                value += alpha[k]*t[k]*kernel(X[:,k],vector,sigma=sigma)

            result[i,j] = value
        print(i)
        
    
    plt.contour(X_grid,Y_grid,result,np.array([-1,0,1]))


    
    plt.scatter(Xnew[0,:],Xnew[1,:], c=tnew)
    fig = plt.gcf()
    ax =fig.gca()
    for i in positions:
        circle = plt.Circle((X[0,i],X[1,i]),0.5,color='g', fill = False)
        ax.add_artist(circle)
    
    plt.show()
    
    return

def error_rate(tpred,target):
    """return the error rate """
    nSamples= np.size(target)
    return np.sum((target-tpred)**2/(4*nSamples))


def plot(X, t, w0, alpha, sv_index, svm):

    grid_size = 100
    sv = alpha[sv_index]
    sv_X = X[sv_index, :]
    sv_T = t[sv_index]

    # draw margin:
    # get min and max values of training dimensions first

    minX = min(X[:, 0])
    maxX = max(X[:, 0])
    minY = min(X[:, 1])
    maxY = max(X[:, 1])

    # create meshgrid for current space
    X1, X2 = np.meshgrid(np.linspace(minX, maxX, grid_size), np.linspace(minY, maxY, grid_size))
    X_new = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])

    # calculate discriminante for all points in space
    Z = svm.discriminant(alpha, w0, X, t, X_new).reshape(X1.shape)

    plt.contour(X1, X2, Z, [0.0], colors='grey')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linestyles='dashed')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linestyles='dashed')

    # show support vectors.
    plt.scatter(sv_X[:, 0], sv_X[:, 1], c="w", marker='.')
    plt.show()

