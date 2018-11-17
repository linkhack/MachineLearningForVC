import LinearBasisFunction as lb

range_v = [0, 5]
#Model = lb.LinearBasisFunction(range_v, 0.1)
Model = lb.LinearBasisFunction(range_v, 0.1)
#print(Model.output_vector)
Model.calculateAndPlot( [[1 ],[1,-1], [1,-2,2 ], ] )


#Model.plotSetup()

#print(len(model.getinput_vector()))
