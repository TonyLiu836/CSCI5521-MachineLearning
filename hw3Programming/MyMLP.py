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
    f_x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f_x

class MLP:
    def __init__(self, num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])
        self.num_hid = num_hid

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        #print('shape of train_x', np.shape(train_x))    #1000x64
        #print('shape of train_y', np.shape(train_y))    #1000x10
        count = 0
        best_valid_acc = 0
        
        #Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations

        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)
            
            z = self.get_hidden(train_x)    
            #print(z)
            #print('shape of z', np.shape(z))    #1000x4
            yPredicted = softmax(np.dot(z, self.weight_2) + self.bias_2)
            #print("yPredicted shape", np.shape(yPredicted))
            #print(yPredicted)
            deltaWeight2 = (train_y - yPredicted)
            #print('shape of deltaWeight2', np.shape(deltaWeight2))
            #print(deltaWeight2)
            deltaWeight1 = np.dot(deltaWeight2, np.transpose(self.weight_2)) * (1 - (z)**2)
            
            #update the parameters based on sum of gradients for all training samples
            
            self.weight_2 = self.weight_2 + lr * np.dot(np.transpose(z), deltaWeight2)
            self.bias_2 = self.bias_2 + lr * np.sum(deltaWeight2, axis = 0)  
            self.weight_1 = self.weight_1 + lr * np.dot(np.transpose(train_x), deltaWeight1)
            self.bias_1 = self.bias_1 + lr * np.sum(deltaWeight1, axis = 0)

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

    def predict(self, x):
        # generate the predicted probability of different classes
        z = self.get_hidden(x) #tanh(np.dot(x, self.weight_1) + self.bias_1)
        yPredicted = softmax(np.dot(z, self.weight_2) + self.bias_2)
        #print(yPredicted)
        #print('shape of x',np.shape(x))
        y = np.zeros([len(x),]) # placeholder

        y = np.argmax(yPredicted, axis = 1)
        #print("shape of y", np.shape(y))
        #print(y)
        return y

    def get_hidden(self, x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        #z = x # placeholder
        
        z = tanh(np.dot(x, self.weight_1) + self.bias_1)
        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
