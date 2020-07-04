'''
Cole Conte CSE 512 Hw 3
https://colab.research.google.com/drive/1Eb-G95_dd3XJ-0hm2qDqdtqMugLkSYE8#scrollTo=erwQND925xoe
https://towardsdatascience.com/how-to-code-gaussian-mixture-models-from-scratch-in-python-9e7975df5252
'''

import argparse
import pandas as pd
import numpy as np
import math

def gmmLearner(train,test,components):
	for digit in range(10):
		trainDigit = train[train.iloc[:,-1]==digit]
		#Does standard normal work for starting params?
		#Scale equally divided among components to start?
		mu = 0.0
		sigmaSq = 1.0
		scale = 1.0
		for row in range(len(trainDigit)):
			#Implement MVN, starting with regular old Normal using col 1
			#for k in range(len(components)):
			x = trainDigit.iloc[row,1]
			likelihood = math.exp(-((x-mu)**2)/(2*sigmaSq))/math.sqrt(2*math.pi*sigmaSq)
			b = likelihood #this will change when we add multiple components
			muold = mu
			mu = ((mu * (row+1)) + x)/(row+2) #THIS ISNT WHAT THE SITE SAYS
			sigmaSq = ((sigmaSq * (row+1)) + ((x-muold)**2))/(row+2) #THIS ISNT WHAT THE SITE SAYS
			#update scale
		print("For digit " + str(digit) + " mu="+str(mu) +" and sigmaSq=" + str(sigmaSq))
	



parser = argparse.ArgumentParser()
parser.add_argument("--components")
parser.add_argument("--train")
parser.add_argument("--test")
args = parser.parse_args()

train = pd.read_csv(args.train,header=None)
test = pd.read_csv(args.test,header=None)
gmmLearner(train,test,int(args.components))


