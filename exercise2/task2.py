# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 17:53:17 2019

@author: arnau
"""
import cvxopt
import numpy as np

#D1= cvxopt.matrix(np.diag(np.ones(5) * -1), tc='d')
#D2= cvxopt.matrix(np.diag(np.ones(5) ), tc='d')
#G = cvxopt.matrix([D1,D2])
#
#print(G)

h1 = cvxopt.matrix(np.zeros(5), tc='d')
h2 = cvxopt.matrix(np.ones(5) * 4, tc='d')
h = cvxopt.matrix([h1,h2])

print(h)