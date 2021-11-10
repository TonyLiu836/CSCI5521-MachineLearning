import numpy as np

def process_data(data, mean=None,std=None):
    # normalize the data to have zero mean and unit variance (add 1e-15 to std to avoid numerical issue)
    e = 1e-15
    if mean is not None:
        # directly use the mean and std precomputed from the training data
        data = (data-mean)/(std+e)
        return data
    else:
        # compute the mean and std based on the training data
        mean = std = 0 # placeholder
        mean = np.mean(data, axis = 0)
        std= np.std(data, axis = 0)
        
        data = (data - mean) / (std + e)
        return data, mean, std

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    for i in range(np.shape(label)[0]):
        one_hot[i,label[i]] = 1
    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    # You may receive some warning messages from Numpy. No worries, they should not affect your final results
    
    f_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) # placeholder
    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    #f_x = np.exp(x) / np.sum(np.exp(x), axis = 1) # placeholder
    
    f_x = np.zeros((np.shape(x)))
    for i in range(len(x)):
        f_x[i] = np.exp(x[i]) / (np.sum(np.exp(x[i])))
    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)


            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters


            #update the parameters based on sum of gradients for all training samples


            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes


        # convert class probability to predicted labels

        y = np.zeros([len(x),]).astype('int') # placeholder

        return y

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        z = x # placeholder

        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
