import LinearBasisFunction as lb

range_v = [0, 5]
#Model = lb.LinearBasisFunction(range_v, 0.1)
Model = lb.LinearBasisFunction(range_v, 0.1)
#print(Model.output_vector)
print(Model.calculateError( [1,1,1 ] ))
print(Model.calculateError( [1,2,3 ] ))
print(Model.calculateError( [0,2,4 ] ))
print(Model.calculateError( [5,4,3 ] ))
print(Model.calculateError( [1,-6,2 ] ))

#Model.plotSetup()

#print(len(model.getinput_vector()))
