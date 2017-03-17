import numpy as np
import scipy.io.arff as sparff
import random
import math 

#Loading data from arff file
def loadData(filename):
    # reading the training data
    data, attribute = sparff.loadarff(filename)
    
    #getting size of attributes and instances
    numFeatures = len(attribute.names())-1
    numInstances = len(data)
    classLabel = attribute[attribute.names()[-1]]
    
    X, Y = [], []
    for i in range(numInstances):
        # convert labels to 0-1 encoding
        Y.append(classLabel[1].index(data[i][-1]))
        
        # create feature vector representation for each isntance
        featureVector = []
        for j in range(numFeatures):
            featureVector.append(data[i][j])
        X.append(featureVector)
        
    npX = np.array(X)
    npY = np.array(Y)
    return npX, npY, attribute, numFeatures, numInstances, classLabel
	

#sigmoid activation function canculation
def sigmoidFunction(input):
    return np.divide(1.0,(np.add(1.0,np.exp(-input))))


#initializing weights for hidden layer, output layer and bias in range of (-0.1,0.1)
def initializaWeights(numFeatures, numHidden):
    
    weightsHidden = []
    weightsOutput = []
    weightsBias = []
    weightsHidden.append(np.random.uniform(-0.1, 0.1, (numFeatures, numHidden)))
    weightsOutput.append(np.random.uniform(-0.1, 0.1, numHidden))
    
    oBias = random.uniform(-0.1,0.1)
    hBias = random.uniform(-0.1,0.1)
    weightsBias.append(oBias)
    weightsBias.append(hBias)
    
    return weightsHidden, weightsOutput, weightsBias
	

#training neural net model using stochastic gradient descend
def trainModel(X, Y, weightsHidden, weightsOutput, weightsBias, foldInstanceIndices, lrate):
    
    randomIndices = np.random.permutation(foldInstanceIndices)
    
    for i in randomIndices:
        
        ## forward pass
        inputRow = np.array(X[i])
        
        ##hidden layer output calculation
        hiddenLayer = []
        for j in range(0,len(weightsHidden)):
            sumWeightsHidden = np.dot(inputRow,weightsHidden[j]) + weightsBias[1]
            hiddenOut = sigmoidFunction(sumWeightsHidden)
            hiddenLayer.append(hiddenOut)

        ##output layer output calculation
        sumWeightsOutput = np.dot(np.array(hiddenLayer),weightsOutput[0]) + weightsBias[0]
        outputOut = sigmoidFunction(sumWeightsOutput)
        
        ##update weights
        weightsHidden, weightsOutput, weightsBias = backProp(inputRow, Y[i], hiddenLayer, outputOut, weightsOutput, weightsHidden, weightsBias, lrate)
        
    return weightsHidden, weightsOutput, weightsBias
	

    
#update weights by backpropagation
def backProp(inputRow, y, hiddenLayer, outputOut, weightsOutput, weightsHidden, weightsBias, lrate):
    ## backpropagation

    #Output layer weight update
    delta_o = outputOut*(1-outputOut)*(y-outputOut)
    #print len(delta_o)
    wtsOut_gradient = np.multiply((lrate*delta_o),hiddenLayer)
    weightsOutput = np.add (weightsOutput, wtsOut_gradient)

    #Hidden layer weight update
    for j in range(0,len(weightsHidden)):
        delta_h = hiddenLayer[j]* (1-hiddenLayer[j])*delta_o*weightsOutput[0][j]
        #print len(delta_h)
        wtsHid_gradient = np.multiply((lrate*delta_h),inputRow)
        weightsHidden [j] =np.add (weightsHidden[j], wtsHid_gradient)
        
    #Bias weights update
    weightsBias[0] = lrate*delta_o
    weightsBias[1] = lrate*delta_h
    
    return weightsHidden, weightsOutput, weightsBias
	

#predict output for a signle instance 
def predictClass(inputRow,weightsHidden,weightsOutput, weightsBias):
    hiddenLayer = []
    for j in range(0,len(weightsHidden)):
        sumWeightsHidden = np.dot(inputRow,weightsHidden[j]) + weightsBias[1]
        hiddenOut = sigmoidFunction(sumWeightsHidden)
        hiddenLayer.append(hiddenOut)

    ##output layer output calculation
    sumWeightsOutput = np.dot(np.array(hiddenLayer),weightsOutput[0]) + weightsBias[0]
    outputOut = sigmoidFunction(sumWeightsOutput)
    return outputOut

	
	
#testing neural net model with learned weights
def testModel(X,Y,weightsHidden,weightsOutput, weightsBias, foldInstanceIndices,fold, classLabel,NNOutput):
    
    for idx in list(foldInstanceIndices):
        
        row = np.array(X[idx])
        output = predictClass(row,weightsHidden,weightsOutput,weightsBias)
        
        if output > 0.5:
            predictedClass = classLabel[1][1]
        else: 
            predictedClass = classLabel[1][0]
        
        if Y[idx]==1:
            actualClass = classLabel[1][1]
        else: 
            actualClass = classLabel[1][0]
            
        if NNOutput.has_key(idx):
            print 'error'
        else:
            NNOutput[idx] = [fold+1, predictedClass, actualClass, output[0]]
    return NNOutput
			

def stratifiedSample(X, Y, nfolds):
    numNegInstance = list(Y).count(0)
    numPosInstance = list(Y).count(1)
    numNegInstance, numPosInstance
    negIndices = np.where(Y==0)[0]
    posIndices = np.where(Y==1)[0]

    posInstanceInFold = (numPosInstance/nfolds)
    negInstanceInFold = (numNegInstance/nfolds)

    foldInstanceIndices = []
    i= j = 0
    for f in range(0,nfolds):
        foldInstanceIndices.append(np.append(posIndices[i:(i+posInstanceInFold)],negIndices[j:(j+negInstanceInFold)]))
        i+=posInstanceInFold
        j+=negInstanceInFold
    
    #distributing rest of the positive examples
    k=0
    for m in range(i, len(posIndices)):
        foldInstanceIndices[k] = np.append(foldInstanceIndices[k], posIndices[m])
    
        if k<nfolds:    
            k+=1
        else:
            k=k%nfolds
    #distributing rest of the negative examples
    k=0        
    for m in range(j, len(negIndices)):
        
        foldInstanceIndices[k] = np.append(foldInstanceIndices[k], negIndices[m])

        if k<nfolds:    
            k+=1
        else:
            k=k%nfolds
    
    return foldInstanceIndices
	

    
def crossValidation(X, Y, foldInstanceIndices, lrate, nfolds, nEpochs, classLabel, numFeatures,NNOutput):
    numHidden = numFeatures
    for f in range(0,nfolds):
        testFold = f
        weightsHidden, weightsOutput, weightsBias = initializaWeights(numFeatures, numHidden)

        for e in range(nEpochs):
            for t in range(0,len(foldInstanceIndices)):
                if t==testFold:
                    continue
                weightsHidden, weightsOutput, weightsBias = trainModel(X,Y,weightsHidden,weightsOutput, weightsBias, foldInstanceIndices[t], lrate)

        NNOutput = testModel(X,Y,weightsHidden,weightsOutput, weightsBias, foldInstanceIndices[testFold],testFold, classLabel, NNOutput)
    return NNOutput


		
def printOutput(numInstances, NNOutput):

    count =0
    tp, tn, fp, fn = 0,0,0,0 
    for i in range(0,numInstances):
        if NNOutput.has_key(i):
            count+=1
            values = NNOutput.get(i)
            print "{} {} {} {}".format(*values)
