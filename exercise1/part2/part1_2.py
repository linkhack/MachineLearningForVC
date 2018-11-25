#import LinearBasisFunction as lb
import ClosedForm as cf

range_v = [0, 5]
#Model = lb.LinearBasisFunction(range_v, 0.1)
#Model = lb.LinearBasisFunction(range_v, 0.1)
#print(Model.output_vector)
#Model.calculateAndPlot( [[1],[1,-1], [1,-2,2 ], [1], [1,0.1,0.1,0.1], ] )
#Model.calculateAndPlot( [[1,-1],[1,0],[1,0.3],[1,0.6],[1,1] ] )

#Model.plotSetup()

#print(len(model.getinput_vector()))

t = cf.ClosedForm([0, 5])
t.presentMode(12)