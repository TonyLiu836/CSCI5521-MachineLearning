
"""
Below is the pseudo code that you may follow to create your python user defined function.

Your function is expected to return :-
    1. final weight vector w
    2. error rate (fraction of training samples that are misclassified)

def keyword can be used to create user defined functions. Functions are a great way to
create code that can come in handy again and again. You are expected to submit a python file
with def MyLeastSquare(X,y) function.

"""

# Header
import numpy as np

# Solve the least square problem by setting gradients w.r.t. weights to zeros
def MyLeastSquare(X,y):
    # placeholders, ensure the function runs
    w = np.array([1.0,-1.0])
    error_rate = 1.0

    # calculate the optimal weights based on the solution of Question 1
    
    XtransXinv = np.linalg.inv(np.dot(np.transpose(X),X))
    w = np.dot(np.dot(XtransXinv, np.transpose(X)), y)
    # compute the error rate
    
    myPrediction = np.dot(X, w)
    numErrors = 0    
    
    #threshold predicted values
    for j in range(np.shape(myPrediction)[0]):
        if myPrediction[j] >= 0:
            myPrediction[j] = 1
        else:
            myPrediction[j] = -1
    
    #compare predicted values with actual values
    for i in range(np.shape(y)[0]):
        if y[i] != myPrediction[i]:
            numErrors += 1

    error_rate = numErrors / (np.shape(y)[0])
    
    return (w,error_rate)
