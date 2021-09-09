"""
Below is the pseudo code that you may follow to create your python user defined function.

Your function is expected to return :-
    1. number of iterations / passes it takes until your weight vector stops changing
    2. final weight vector w
    3. error rate (fraction of training samples that are misclassified)

def keyword can be used to create user defined functions. Functions are a great way to
create code that can come in handy again and again. You are expected to submit a python file
with def MyPerceptron(X,y,w) function.

"""
# Hints
# one can use numpy package to take advantage of vectorization
# matrix multiplication can be done via nested for loops or
# matmul function in numpy package


import numpy as np

# Implement the Perceptron algorithm
def MyPerceptron(X,y,w0=[1.0,-1.0]):
    k = 0 # initialize variable to store number of iterations it will take
          # for your perceptron to converge to a final weight vector
    w=w0
    error_rate = 1.00
    wPrev = [3,3]
    
    # loop until convergence (w does not change at all over one pass)
    # or until max iterations are reached
    # (current pass w ! = previous pass w), then do:
    #
    while(w != wPrev).all():

        # for each training sample (x,y):
            # if actual target y does not match the predicted target value, update the weights
        wPrev = w
        k += 1
        for i in range(np.shape(X)[0]):
            if np.dot(y[i], np.dot(X[i], w)) <= 0:
                w = w + np.dot(y[i], X[i])
    
            # calculate the number of iterations as the number of updates

    myPrediction = np.dot(X, w)
    
    #threshold predicted values
    for j in range(np.shape(myPrediction)[0]):
        if myPrediction[j] >= 0:
            myPrediction[j] = 1
        else:
            myPrediction[j] = -1

    # make prediction on the csv dataset using the feature set
    # Note that you need to convert the raw predictions into binary predictions using threshold 0

    numErrors = 0    
    
    #compare predicted values to actual values
    for n in range(np.shape(y)[0]):
        if y[n] != myPrediction[n]:
            numErrors += 1
    
    error_rate = numErrors / (np.shape(y)[0])
    
    # compute the error rate
    # error rate = ( number of prediction ! = y ) / total number of training examples

    return (w, k, error_rate)
