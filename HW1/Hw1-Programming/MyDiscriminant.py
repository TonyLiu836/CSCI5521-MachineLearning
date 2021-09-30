import numpy as np

class GaussianDiscriminant:
    def __init__(self,k=2,d=8,priors=None,shared_cov=False):
        self.mean = np.zeros((k,d)) # mean
        self.shared_cov = shared_cov # using class-independent covariance or not
        if self.shared_cov:
            self.S = np.zeros((d,d)) # class-independent covariance
        else:
            self.S = np.zeros((k,d,d)) # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class

        class1x = []
        class2x = []
        
        for i in range(np.shape(ytrain)[0]):
            if ytrain[i] == 1:
                class1x.append(Xtrain[i,:])
            else:
                class2x.append(Xtrain[i,:])
        
        class1x = np.asarray(class1x)
        class2x = np.asarray(class2x)

        m1 = np.sum(class1x, axis=0) / self.d
        m2 = np.sum(class2x, axis=0) / self.d

        self.mean[0,:] = m1
        self.mean[1,:] = m2
        
        S1 = np.zeros((self.d, self.d))
        S2 = np.zeros((self.d, self.d))
        
        if self.shared_cov:
            # compute the class-independent covariance
            '''
            for i in range(self.d):
                for j in range(self.d):
                    numerator = 0
                    for N in range(np.shape(class1x)[0]):
                        p1 = (class1x[N,i] - self.mean[0,i])
                        p2 = (class1x[N,j] - self.mean[0,j])
                        numerator += p1*p2

                    S1[i,j] = numerator / np.shape(class1x)[0]

            for m in range(self.d):
                for n in range(self.d):
                    numer = 0
                    for N in range(np.shape(class2x)[0]):
                        numer += (class2x[N,m] - self.mean[1,m]) * (class2x[N,n] - self.mean[1,n])
                    S2[m,n] = numer / np.shape(class2x[0])
            self.S = self.p[0]*S1 + self.p[1]*S2
            '''
            self.S = np.cov(Xtrain, rowvar = False, ddof=0)
            #pass # placeholder
        else:
            # compute the class-dependent covariance
            
            S1 = np.cov(class1x,rowvar = False, ddof=0)
            S2 = np.cov(class2x,rowvar = False, ddof=0)
            self.S[0,:,:] = S1
            self.S[1,:,:] = S2
            '''
            for i in range(self.d):
                for j in range(self.d):
                    numerator = 0
                    #if i == j:
                    for N in range(np.shape(class1x)[0]):
                        p1 = (class1x[N,i] - self.mean[0,i])
                        p2 = (class1x[N,j] - self.mean[0,j])
                        numerator += p1*p2

                    self.S[0,i,j] = numerator / np.shape(class1x)[0]
            
            for m in range(self.d):
                for n in range(self.d):
                    numer = 0
                    for N in range(np.shape(class2x)[0]):
                        numer += (class2x[N,m] - self.mean[1,m]) * (class2x[N,n] - self.mean[1,n])
                    self.S[1,m,n] = numer / np.shape(class2x[0])
           '''
        #print("shape of S=", np.shape(self.S))
            #pass

    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder
        g = np.zeros(self.k)
        for i in np.arange(Xtest.shape[0]): # for each test set example

            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                if self.shared_cov:
                    
                    w = np.dot(np.linalg.inv(self.S), self.mean[c,:])
                    
                    w0 = -(1/2) * np.dot(np.dot(np.transpose(self.mean[c,:]), np.linalg.inv(self.S)), self.mean[c,:]) + np.log(self.p[c])
                    
                    #print("w=", w)
                    #print("w0=", w0)
                    
                    g[c] = np.dot(np.transpose(w), Xtest[i,:]) + w0
                    
                    #pass # placeholder
                else:
                    '''
                    print("np.transpose(self.mean[c,:]) shape=", np.transpose(self.mean[c,:]))
                    print("np.linalg.inv(self.S[c,:,:]) shape=", np.shape([np.linalg.inv(self.S[c,:,:])]))
                    print("log(p) shape", np.shape([np.log(self.p)]))
                    print("shape of X[i,:]", np.transpose(Xtest[i,:]))
                    '''
                    
                    W = (-1/2) * (np.linalg.inv(self.S[c,:,:]))
                    #print("W", W)
                    w = np.dot(np.linalg.inv(self.S[c,:,:]), self.mean[c,:])
                    #print("np.log(np.absolute(self.S[c,:,:])) = ", np.shape(np.log([np.absolute(self.S[c,:,:])])))
                    #w0 = (-1/2) * np.transpose(self.mean[c,:]) * np.linalg.inv(self.S[c,:,:]) * self.mean[c,:] - (1/2) * np.log(np.absolute(self.S[c,:,:])) * np.log(self.p)
                    #print("log det of S", np.log(np.linalg.det(self.S[c,:,:])))
                    #print("w0 first part", np.dot(np.transpose(self.mean[c,:]), np.linalg.inv(self.S[c,:,:])))
                    
                    
                    
                    w0 = (-1/2) * np.dot(np.dot(np.transpose(self.mean[c,:]), np.linalg.inv(self.S[c,:,:])), self.mean[c,:]) - (1/2) * np.log(np.linalg.det(self.S[c,:,:])) + np.log(self.p[c])
                    #print(w0)
                    #print("shape of matrix multiplication", np.dot(np.dot(Xtest[i,:], W), np.transpose(Xtest[i,:])))# + np.transpose(w) * Xtest[i,:] + w0)
                    #print("shape of W", np.shape(W))
                    #print("shape of w", np.shape(w))
                    #print("w0", np.shape(w0))
                    
                    #print("w0", w0)
                    
                    g[c] = np.dot(np.dot(Xtest[i,:], W), np.transpose(Xtest[i,:])) + np.dot(np.transpose(w), Xtest[i,:]) + w0
                    #pass
                
            # determine the predicted class based on the values of discriminant function
            if g[0] > g[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        if self.shared_cov:
            return self.mean[0], self.mean[1], self.S
        else:
            return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Diagonal:
    def __init__(self,k=2,d=8,priors=None):
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,)) # variance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d
        
        
    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        
        class1x = []
        class2x = []
        
        for i in range(np.shape(ytrain)[0]):
            if ytrain[i] == 1:
                class1x.append(Xtrain[i,:])
            else:
                class2x.append(Xtrain[i,:])
        #print("before asarray=",class1x)
        class1x = np.asarray(class1x)
        class2x = np.asarray(class2x)
        #print("after asarray=",class1x)
        m1 = np.zeros((1,self.d)) 
        m2 = np.zeros((1,self.d)) 
        
        m1 = np.sum(class1x, axis=0) / self.d
        m2 = np.sum(class2x, axis=0) / self.d
        self.mean[0,:] = m1
        self.mean[1,:] = m2
        # compute the variance of different features
        
        #for c in range(self.d):
            
        
        pass # placeholder

    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        for i in np.arange(Xtest.shape[0]): # for each test set example
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                pass

            # determine the predicted class based on the values of discriminant function

        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
