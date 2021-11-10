#import libraries
import numpy as np

class Kmeans:
    def __init__(self,k=8): # k is number of clusters
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        # initialize the centers of clusters as a set of pre-selected samples
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005] # indices for the samples
        num_iter = 0 # number of iterations for convergence
        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False
        #print("vertical shape of X",np.shape(X)[0])
        if np.shape(X)[0] != 1:
            self.center = X[init_idx, :]
        else:
            self.center = X[init_idx]
        
        # iteratively update the centers of clusters till convergence
        while not is_converged:
            datainCluster = {}
            for n in range(self.num_cluster):
                datainCluster[n] = [] 
            tally = np.zeros((3,8))
            dist = [0,0,0,0,0,0,0,0]#np.zeros(len(init_idx))
            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                for j in range(self.num_cluster):
                    dist[j] = np.linalg.norm(X[i] - self.center[j])**2 
                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                minIndex = dist.index(min(dist))
                cluster_assignment[i] = minIndex
                datainCluster[minIndex].append([X[i]])
                
                if y[i] == 0:
                    tally[0,minIndex] += 1
                elif y[i] == 8:
                    tally[1,minIndex] += 1
                else:
                    tally[2,minIndex] += 1

            for d in range(self.num_cluster):
                self.center[d] = np.average(datainCluster[d], axis = 0)   
            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)
            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1
            
        # compute the information entropy for different clusters
        entropy = float('inf') # placeholder
        e = 10e-15
        H = np.zeros((8,))
        for k in range(self.num_cluster):
            for c in range(np.shape(tally)[0]):
                total = np.sum(tally[:,k])
                H[k] += tally[c,k]/total * np.log2(tally[c,k]/total + e)
        H = -H
        #print(H)
        entropy = np.average(H)
        return num_iter, self.error_history, entropy

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for i in range(len(X)):
            error += np.linalg.norm(X[i] - self.center[cluster_assignment[i]])**2
        return error

    def params(self):
        return self.center
