import exercise1.part1.PerceptronOnlineTraining as online_perceptron
import exercise1.part1.Perceptron_batch_test as batch_perceptron
import matplotlib.pyplot as plt
import numpy as np


def perc(weights, data):
    """Assumes the data was already augmented (homogeneous coordinates) training_set[0,:]=1"""
    result = np.sign(np.dot(weights, data))  # @ is matrix multiplication
    return result


def percTrain(data, targets, maxIts, online):
    """Assumes the data was already augmented (homogeneous coordinates) training_set[0,:]=1"""
    if online:
        return online_perceptron.perceptron_online_training(data, targets, max_iterations=maxIts)
    else:
        return batch_perceptron.percTrain(data, targets, maxIts)


def plot_decision_boundary(weights,data,targets, transformed):
    dim_data = np.size(data, 0)
    features = data[1:2, :]
    min_x = np.min(data[1,:])
    max_x = np.max(data[1,:])

    min_y = np.min(data[2,:])
    max_y = np.max(data[2,:])

    x_axis = np.linspace(min_x,max_x,500)
    y_axis = np.linspace(min_y,max_y,500)

    X_grid, Y_grid = np.meshgrid(x_axis,y_axis);
    if transformed:
        transformed_features = np.array(([np.ones((500,500)),X_grid,Y_grid, X_grid*X_grid,Y_grid*Y_grid, X_grid*Y_grid]))
    else:
        # transformed_features = np.array([[[1,X_grid[i,j], Y_grid[i,j]] for j in range(500)] for i in range(500)])
        transformed_features = np.array(([np.ones((500,500)),X_grid,Y_grid]))
    
    result = np.sum(np.array([weights[k]*transformed_features[k,:,:] for k in range(0,dim_data)]),0) #sum along the 0_dimension

    fig = plt.contour(X_grid,Y_grid,result,0);
    plt.scatter(data[1,:],data[2,:], c = targets)
#    fig.show()

    return fig

def plot_weights_full(weights):
    """plot the image of the weights for the full image perceptron"""
    naugmented_weights =  weights[1:]
    
    x_axis = np.arange(0,28)
    y_axis = np.arange(27,-1,-1)
    
    X_grid, Y_grid = np.meshgrid(x_axis,y_axis)
    
    X_grid = np.reshape(X_grid,[784,1])
    Y_grid = np.reshape(Y_grid,[784,1])
    
    print(np.shape(naugmented_weights))
    
    plt.scatter(X_grid,Y_grid)#, c = naugmented_weights)
    
    
def confusion_matrix(digit,predicted_labels,labels):
    """plot the confusion matrix of an experiment """
    nbr_of_data = np.size(labels)
    
    Correct_value_class1 = 0 
    False_value_class1 = 0
    nbr_class1 = 0
    
    Correct_value_class2 = 0 
    False_value_class2 = 0
    nbr_class2 = 0
    
    for i in range(0,nbr_of_data):
        predicted_value = predicted_labels[i]
        value = labels[i]
        
        if predicted_value == value:
            if value == 1:
                Correct_value_class1 +=1
                nbr_class1 += 1
            else:
                Correct_value_class2 +=1
                nbr_class2 += 1
        else:
            if value == 1:
                False_value_class1 += 1
                nbr_class1 += 1
            else:
                False_value_class2 += 1
                nbr_class2 += 1
                
    perc_Correct1 = Correct_value_class1/nbr_class1
    perc_false1 =   False_value_class1/nbr_class1
    
    perc_Correct2 = Correct_value_class2/nbr_class2
    perc_false2 =   False_value_class2/nbr_class2
    
#    x1 = [perc_Correct1,perc_Correct2]
#    x2 = [perc_false1,perc_false2]
#    bins =[0.5,1.5]
#    
#    plt.hist([x1,x2], bins = bins, color = ['green','red'], label=['Correct','Wrong'])
#    plt.xlabel('class')
#    plt.ylabel('percentage')
#    
#    plt.show()
      
    
    
    




