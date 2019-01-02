import matplotlib.pyplot as plt
import matplotlib.image as pltImg
import numpy as np
import exercise2.Kernel as kernel

def plot_SVM(alpha,w0,X,t,Xnew,tnew,kernel=kernel.linearkernel):
    """ ploting of the decision boundaries and the margin for the SVM trained on (X,t) 
    assuming that X represent vector written by colums"""
    nFeatures, nSamples = X.shape
    min_x = np.min(Xnew[1,:])
    max_x = np.max(Xnew[1,:])

    min_y = np.min(Xnew[2,:])
    max_y = np.max(Xnew[2,:])
    
    x_axis = np.linspace(min_x,max_x,500)
    y_axis = np.linspace(min_y,max_y,500)
    
    X_grid, Y_grid = np.meshgrid(x_axis,y_axis)
    
    result = np.zeros([500,500])
    for i in range(0,500):
        for j in range(0,500):
            vector = np.array([[X_grid[i,j]],[Y_grid[i,j]]])
            value = w0+np.sum(np.array([alpha[k]*t[k]*kernel(X[:,k],vector) for k in range(0,nSamples)]))
            result[i,j] = value
    
    plt.contour(X_grid,Y_grid,result,0)
    plt.contour(X_grid,Y_grid,result,1)
    plt.contour(X_grid,Y_grid,result,-1)
    
    plt.scatter(Xnew[0,:],Xnew[1,:], c=tnew)
    
    plt.show()
    
    return
    