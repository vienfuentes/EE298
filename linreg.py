import argparse
import sys
import numpy as np
import random

# parser
def int_args(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('integers',
							metavar='N',
							type=int,
							nargs='+',
							help='integer args')
	
	return parser.parse_args()

# create polynomial from parsed arguments
p = np.poly1d(int_args(sys.argv[1:]).integers)

def cost_function(loss):
	m = len(loss)
	total_loss = 0
	for i in range (0, len(loss)):
		total_loss += loss[i] * loss[i]
	cost = total_loss / 2 / m
	return cost

def hypothesis_function(x, theta):
	hypothesis = np.dot(x, theta)
	return hypothesis

def loss_function(hypothesis, y):
	loss = hypothesis - y
	return loss

def gradient_function(xTrans, loss, m):
	gradient = np.dot(xTrans, loss) / m
	return gradient

def gradientDescent(x, y, theta, alpha, m, iterations):
	for i in range (0, iterations):
		hypothesis = hypothesis_function(x, theta)
		loss = loss_function(hypothesis, y)
		cost = cost_function(loss)
		print("Itr.: %d | Cost: %f | %s" % (i, cost, theta[::-1]))
		gradient = gradient_function(x.transpose(), loss, m)
		theta = theta - alpha * gradient
	return theta

def generateDataPoints(num, poly, noiseAmp):
	# create num amount of data points
	# create x with size based on the amount of arguments/polynomials
	l = len(int_args(sys.argv[1:]).integers)
	x = np.zeros(shape=(num, l))
	y = np.zeros(shape=num)
	# make the data small to prevent overflow
	amp = 1 / 100
	# basically a linear distribution for the data points, could be modified to be something else
	for i in range (0, num):
		i_amp = i * amp
		for j in range (0, l):
			x[i][j] = (i_amp) ** (j)
		# uniform noise added
		y[i] = poly(i_amp) + random.uniform(0, 1) * noiseAmp
	return x, y

# noise added has a value smaller than the data generated to be not too noisy
noiseAmp = 1 / 10000
# the current alpha used is 0.01, however, this code was tested from 0.01 to 0.0005, 
# with results in the acceptable range, GIVEN a certain range of other parameters.
# other parameters modified were the amount of data points (should not be too few), and 
# the amount of iterations (higher values could ideally could lower cost to 0, given enough time)
# polynomial coefficients greater than 20 possibly could have inaccurate results; this could be
# remedied by increasing the amount of iterations or the learning rate.
# for the sake of speeding up runtime, the amount of iterations (and other corresponding parameters)
# were left as seen here in the code
x, y = generateDataPoints(250, p, noiseAmp)
m, n = np.shape(x)
iterations = 100000
alpha = 0.01
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, iterations)
# just flip the final array
theta = theta[::-1]
print ("")
print ("Input arguments / result of numpy.poly1d (in the 2 succeeding lines):")
print (p)
print ("")
print ("Computed coefficients of the polynomial (highest order leftmost)")
print (theta)
print ("")
print ("Rounding off")
print (theta.round())
