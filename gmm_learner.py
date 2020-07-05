'''
Cole Conte CSE 512 Hw 3
https://colab.research.google.com/drive/1Eb-G95_dd3XJ-0hm2qDqdtqMugLkSYE8#scrollTo=erwQND925xoe
https://towardsdatascience.com/how-to-code-gaussian-mixture-models-from-scratch-in-python-9e7975df5252
'''

import argparse
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
import math

def gmmLearner(train,test,components):
	#Training
	for digit in range(10):
		#Subset dataframe
		trainDigit = train[train.iloc[:,-1]==digit]
		x = trainDigit.iloc[:,:-1]
		#x = x.astype(float)

		#Initialization Step
		#Start with each component scaled equally
		scales = [1.0/ components]*(components)
		#mus start as mean of x
		mus = np.asarray(x.mean(axis=0))
		#Covariance matrix starts as cov matrix of x
		cov = np.cov(x.T)

		#Run EM algo 50 times
		for i in range(50):
			#E Step
			likelihood = multivariate_normal.pdf(x=x, mean=mus, cov=cov,allow_singular=True)
			#M Step
			postProb = likelihood
			mus = np.sum(postProb.reshape(len(x),1) * x, axis=0) / (np.sum(postProb))
			cov = np.dot((postProb.reshape(len(x),1) * (x - mus)).T, (x - mus)) / (np.sum(postProb))
			scales = np.mean(postProb)
		print("For digit "  + str(digit) + " mu=\n"+str(mus) +"\n and cov=\n" + str(cov))

		
	
	#Test




parser = argparse.ArgumentParser()
parser.add_argument("--components")
parser.add_argument("--train")
parser.add_argument("--test")
args = parser.parse_args()

train = pd.read_csv(args.train,header=None)
test = pd.read_csv(args.test,header=None)
gmmLearner(train,test,int(args.components))


