from nn import *
import sys
import numpy as np

def getInputArgs():
    # process input arguments
	
	if len(sys.argv)== 5:
		fname_train = str(sys.argv[1])
		numFold = int (str(sys.argv[2]))
		learningRate = float (str(sys.argv[3]))
		numEpoch = int (str(sys.argv[4]))

	else:
		print len(sys.argv)
		raise ValueError('This program takes input arguments as: '
				 '\n\tneuralnet trainfile num_folds learning_rate num_epochs\n')
	return fname_train, numFold, learningRate, numEpoch 

NNOutput = {}
# load data
filename, numFold, learningRate, numEpoch = getInputArgs()
X, Y, attribute, numFeatures, numInstances, classLabel = loadData(filename)

#create stratified sample
foldInstanceIndices = stratifiedSample(X, Y, numFold)

#train neural net with cross validation
NNOutput = crossValidation(X, Y, foldInstanceIndices, learningRate, numFold, numEpoch, classLabel, numFeatures, NNOutput)

#print output 
printOutput(numInstances, NNOutput)
