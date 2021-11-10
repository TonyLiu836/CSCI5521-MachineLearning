import numpy as np

def PCA(X,num_dim=None):
    X_pca, num_dim = X, num_dim #len(X[0]) # placeholder
    mean = np.mean(X,axis=0)
    Xcentered = X - mean
    cov = np.cov(np.transpose(Xcentered))
    
    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
    
    lamb, vect = np.linalg.eigh(cov)
    #print("shape of lambda",len(lamb))
    #print("shpae of eigenvect matrix", np.shape(vect))
    
    #PCAs = []
    #lamb = lamb.sort()
    #print("lambda", lamb)
    # select the reduced dimensions that keep >95% of the variance
    #print("bool of num_dim", num_dim is None)
    #print("num_dim=", num_dim)
    #print("vect[0][0]",vect[0][0])
    if num_dim is None:
        #print("code reaches here")
        diagSum= 0
        var = 0
        for i in range(len(lamb)):
            diagSum += lamb[i]
        
        dim = 0
        idx = lamb.argsort()[::-1]
        lamb = lamb[idx]
        vect = vect[:,idx]
        ratio = var / diagSum
        while ratio < .95:
            #for i in range(np.shape(lamb)[0]):
            #    var += vect[i][i]
            #    var = var/diagSum
            var += lamb[dim]
            ratio = var/diagSum
            dim += 1
            #print(var)
            #var = var/diagSum 
        #print("index=",dim)
        #print("var=", var)
        #print("diagSum", diagSum)
        w = vect[:,0:dim]
        #print("shape of w", np.shape(w))
        X_pca = np.dot(Xcentered, w)
        num_dim = dim
    # project the high-dimensional data to low-dimensional one
    else:
        maxEigValIndex = np.argmax(lamb)
        #print("shape of eigenvect matrix", np.shape(vect))
        w = vect[:,maxEigValIndex]
        #print("shape of w", np.shape(w))
        X_pca = np.dot(Xcentered, np.transpose([w]))
        #print(np.shape(X_pca))
        #X_pca = np.transpose([X_pca])
    return X_pca, num_dim
