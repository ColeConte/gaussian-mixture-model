'''
Cole Conte CSE 512 Hw 3
'''

import argparse
import pandas as pd
import numpy as np
import random
from scipy.stats import multivariate_normal
import math

def gmmLearner(train,test,components):
	#Training
	models = []
	random.seed(123)
	for digit in range(10):
		#Subset dataframe
		trainDigit = train[train.iloc[:,-1]==digit]
		x = trainDigit.iloc[:,:-1]

		#Initialization Step
		#Add clusters
		for j in range(components):
			x["c"+str(j)] = 0
		
		#Initialize into random cluster
		randClusters = [random.randint(0,components-1) for _ in range(len(x))]
		x["cluster"] = randClusters
		for j in range(components):
			x["c"+str(j)] = x["cluster"].map(lambda y: 1.0 if y==j else 0.0)
		x = x.drop(["cluster"],axis=1)



		#Mus start as mean of cluster
		#Covariance matrix starts as cov matrix of cluster
		mus = []
		covs = []
		for j in range(components):
			cluster = x[x["c"+str(j)] == 1].iloc[:,:-components]
			mus+=[(np.asarray(cluster.mean(axis=0)))]
			covs+=[(np.cov(cluster.T))]
		

		#Start with each component scaled equally
		weights = [1.0/ components]*(components)

		#Run EM algo 20 times
		for i in range(20):
			#E Step
			likelihood = []
			sumLikWeight = np.zeros((1,len(x)))
			for j in range(components):
				likelihood += [multivariate_normal.pdf(x=x.iloc[:,:-components], mean=mus[j], cov=covs[j],allow_singular=True)]
				sumLikWeight = np.add(sumLikWeight,[likelihood[j]*weights[j]])

			#M Step
			postProb = []
			for j in range(components):
				postProb += [likelihood[j]*weights[j] / sumLikWeight]
			for j in range(components):
				for m in range(len(x.columns)-components):
					postProbSum = np.sum(postProb[j])
					xVals = x.iloc[:,m].values.flatten()
					mus[j][m] = (np.dot(postProb[j], xVals))/postProbSum
				weightedx = (x.iloc[:,:-components]-mus[j])
				for i in range(len(x)):
					weightedx.iloc[i,:] = (postProb[j][0][i] * weightedx.iloc[i,:]).T
				xMinusMu = (x.iloc[:,:-components]-mus[j])
				xMinusMu.columns = weightedx.columns
				covs[j] = np.dot(xMinusMu.T, weightedx) /postProbSum
				weights[j] = np.mean(postProb[j])
		mixedModel = []
		for j in range(components):
			mixedModel += [{"mus":mus[j],"cov":covs[j],"weight":weights[j]}]
		models += [mixedModel]

	#Testing
	test.columns = ["Col"+str(x) for x in range(len(test.columns)-1)] + ["Y"]
	for digit in range(10):
		#Remove correct label and previously computed probs
		x = test.iloc[:,:-(1+digit)]
		prob = 0
		for j in range(components):
			prob += (multivariate_normal.pdf(x=x, mean=models[digit][j]['mus'], cov=models[digit][j]['cov'],allow_singular=True))*models[digit][j]['weight']
		test[str(digit)] = prob
	test["Prediction"] = test.iloc[:,-10:].idxmax(axis=1)
	#recast
	test["Prediction"] = pd.to_numeric(test["Prediction"])

	#Results Printout
	for digit in range(10):
		testDigit = test[test["Y"]==digit]
		print("Test error for digit " +str(digit) + " is: " + str(1-(len(testDigit[testDigit["Y"]==testDigit["Prediction"]])/float(len(testDigit)))))
	print("Overall test error: "+ str(1-(len(test[test["Y"]==test["Prediction"]])/float(len(test)))))


parser = argparse.ArgumentParser()
parser.add_argument("--components")
parser.add_argument("--train")
parser.add_argument("--test")
args = parser.parse_args()

train = pd.read_csv(args.train,header=None)
test = pd.read_csv(args.test,header=None)
gmmLearner(train,test,int(args.components))


