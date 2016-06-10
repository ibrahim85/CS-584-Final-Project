from __future__ import print_function
import numpy as np
import math
import scipy as sp
import time
import math
import SVM
#from sknn.mlp import Classifier,Layer


def CrossValidation(K,X,gamma, c):  #crossValidation m=1 then use explicit fit, m=2 then use gradient descent fit
    print(X.shape[0])
    classes=[]
    classNum=0
    for y in X[:,-1]:
        if not y in classes:
            classes.append(y)
            classNum+=1

    precisions=np.zeros(shape=(classNum)).tolist()
    recalls=np.zeros(shape=(classNum)).tolist()
    accuracys=0
    fMeasures=np.zeros(shape=(classNum)).tolist()
    for k in xrange(K):
        #for k in range(0,1):
         training=np.ndarray(shape=(0,X.shape[1]))
         validation=np.ndarray(shape=(0,X.shape[1]))

         for i in range(0,X.shape[0]):
              if i % K!=k:
                 training=np.vstack([training,X[i]])
              else:
                 validation=np.vstack([validation,X[i]])

         yExpected=validation[:,-1]
         xs=validation[:,0:validation.shape[1]-1]

         classifier=SVM.svm(training[:,0:X.shape[1]-1],training[:,X.shape[1]-1],gamma=gamma,c=c)
         classifier.train()

         confusionMatrix=np.zeros(shape=(classNum,classNum),dtype=float)

         count=0
         for x in xs:
              #print(classifier.predict(x))
              j=classifier.predict(x)
              confusionMatrix[j,yExpected[count]]=confusionMatrix[j,yExpected[count]]+1
              count+=1

            #confusionMatrix[classes.index(ys[count]),classes.index(y)]=confusionMatrix[classes.index(ys[count]),classes.index(y)]+1

         print(confusionMatrix)
         precision=np.zeros(shape=(classNum),dtype=float)
         recall=np.zeros(shape=(classNum),dtype=float)
         accuracy=0
         fMeasure=np.zeros(shape=(classNum),dtype=float)
         for i in range(classNum):
             if np.sum(confusionMatrix[i,:])==0:
                 precision[i]=0
             else:
                 precision[i]=confusionMatrix[i,i]/np.sum(confusionMatrix[i,:])
             if np.sum(confusionMatrix[:,i])==0:
                 recall[i]=0
             else:
                 recall[i]=confusionMatrix[i,i]/np.sum(confusionMatrix[:,i])
             accuracy+=confusionMatrix[i,i]
             if precision[i]==0 or recall[i]==0:
               fMeasure[i]=0
             else:
               fMeasure[i]=2*precision[i]*recall[i]/(precision[i] +recall[i])
         accuracy=accuracy/validation.shape[0]

         precisions=precisions+precision
         recalls=recalls+recall
         accuracys=accuracys+accuracy
         fMeasures=fMeasures+fMeasure


    p=np.array(precisions)/K
    r=np.array(recalls)/K
    a=accuracys/K
    f=np.array(fMeasures)/K
    print("precision:")
    print(p)
    print("recall:")
    print(r)
    print("accuracy:")
    print(a)
    print("F measure:")
    print(f)



test_x = [[1,2,3], [9,8,7], [1,3,4], [1,3,2], [8,9,10], [8,9,7]]
text_y = [1, -1, 1, 1, -1, -1]


test_all = np.array([[1,2,3,1], [9,8,7,-1], [1,3,4,1], [1,3,2,1], [8,9,10,-1], [8,9,7,-1]])
print (test_all.shape)
CrossValidation(2, test_all, gamma=1, c=1)

#test.train()
print("---------------------------")
#x1 = np.array([1, 2, 3])
#x1_prime = test.linear_scale(x1)

#x2 = np.array([9, 8, 7])
#x2_prime = test.linear_scale(x2)

#x3 = np.array([2, 2, 1])
#x3_prime = test.linear_scale(x3)