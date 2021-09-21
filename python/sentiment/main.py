import numpy
import math

def sigmoid(x):
	return 1 / (1 + math.exp(-x))


w=[2.5,-5,-1.2,0.5,2,0.7]
x=[3,2,1,3,0,4.19]
b=0.1

print(sigmoid(numpy.dot(w,x)+b))
