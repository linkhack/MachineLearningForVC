import matplotlib.pyplot as plt
import matplotlib.image as pltImg
import numpy as np
import exercise2.Kernel as kernel

def plot_SVM(alpha,w0,positions,X,t,Xnew,tnew,kernel=kernel.linearkernel):
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
            vector = np.array([[X_grid[i,j]],[Y_grid[i,j]]])
            value = w0
            for k in range(nSamples):
                value += alpha[k]*t[k]*kernel(X[:,k],vector)

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
    