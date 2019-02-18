import numpy as np
import timeit

def outer_indexing():
    matrix = np.random.rand(10500,10500)
    index = np.random.choice(range(10500),70,replace=False)
    logical_index = np.full((10500,1),False)
    logical_index[index] = True
    return matrix[np.outer(index,index)]

def direct_indexing():
    matrix = np.random.rand(10500,10500)
    index = np.random.choice(range(10500),70,replace=False)
    logical_index = np.full((10500,1),False)
    logical_index[index] = True
    return matrix[index,:][:,index]

def main():

    #result1 = timeit.Timer(outer_indexing).repeat(1)
    result2 = timeit.Timer(direct_indexing).repeat(1)
    print(min(result2))
    print("hallo")

if __name__ == '__main__':
    main()