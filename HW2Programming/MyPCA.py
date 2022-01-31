import numpy as np

def PCA(X,num_dim=None):
    X_pca, num_dim = X, num_dim #len(X[0]) # placeholder
    mean = np.mean(X,axis=0)
    Xcentered = X - mean
    cov = np.cov(np.transpose(Xcentered))
    
    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
    
    lamb, vect = np.linalg.eigh(cov)

    if num_dim is None:
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
            var += lamb[dim]
            ratio = var/diagSum
            dim += 1

        w = vect[:,0:dim]

        X_pca = np.dot(Xcentered, w)
        num_dim = dim
        
    # project the high-dimensional data to low-dimensional one
    else:
        maxEigValIndex = np.argmax(lamb)
        
        w = vect[:,maxEigValIndex]
        
        X_pca = np.dot(Xcentered, np.transpose([w]))
        
    return X_pca, num_dim
