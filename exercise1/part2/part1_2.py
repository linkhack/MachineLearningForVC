import LMS as lms
import ClosedForm as cf
import random

########## 1.2.1

range_v = [0, 5]
lms = lms.LMS(range_v, 0.1)
# init vector for w, and plot flag

#random start w:
w = []
for y in range(3):
    val = round(random.random(), 1)
    w.append(val)


print('starting w :'+str(w))


# 1000 iterations,
# 0.001 as gamma
# plot output (every 75th iteration)
result = lms.learn( w , 1000, 0.001 , 1 )

# 500 iterations,
# 0.01 as gamma
# plot output (every 75th iteration)
#result = lms.learn( [0,0,0] , 500, 0.009 , 1 )

# 100 iterations,
# 0.1 as gamma
# plot output (every 75th iteration)
#result = lms.learn( [0,0,0] , 500, 0.09 , 1 )


# used closed form 
t = cf.ClosedForm([0, 5])
t.setSetup(lms.getSetup())
result_c = t.calcOptimalW(2)
print('-----------------------')
print("lms error: " + str(result[0]))
print("cf error: " + str(result_c[0]))
print('-----------------------')
print("lms w: " + str(result[1]))
print("cf w: " + str(result_c[1]))
print('-----------------------')

#show how overfitting impacts results
#t.presentMode(9)