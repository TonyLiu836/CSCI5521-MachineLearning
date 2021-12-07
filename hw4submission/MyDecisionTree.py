import numpy as np

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.is_leaf = False # whether or not the current node is a leaf node
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node)
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy   # min node entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        #print("prediction shape", np.shape(prediction))    #3000 vector
        #print('text_x shape', np.shape(test_x))            #3000 x 64
        node = self.root
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample
            #pass # placeholder
            data = test_x[i,:]
            #print('data shape',np.shape(data))
            #print('data[3]', data[3])
            #print(data)
            while node:
                #print(node.feature)
                print('data[node.feature]', data[node.feature])
                print('data[node.feature] shape', np.shape(data[node.feature]))
                
                if data[node.feature] == 0:
                    node = node.left_child
                else:
                    node = node.right_child
                #node = node.left_child
                #node = node.right_child
            
            prediction[i] = node.label
        return prediction

    def generate_tree(self,data,label):
        # initialize the current tree node
        #print('data shape', np.shape(data))   #3000 x 64
        
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)

        # determine if the current node is a leaf node
        if node_entropy < self.min_entropy:
            # determine the class label for leaf node
            #cur_node.isleaf = True
            self.label = label
            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature

        # split the data based on the selected feature and start the next level of recursion
        featureData = data[:,selected_feature]
        zeros = np.asarray(np.where(featureData == 0))
        ones = np.asarray(np.where(featureData == 1))
        leftData = data[zeros[0]]
        leftLabel = label[zeros[0]]
        
        rightData = data[ones[0]]
        rightLabel = label[ones[0]]
        
        #leftEntropy = self.compute_node_entropy(leftLabel)
        #rightEntropy = self.compute_node_entropy(rightLabel)
        
        cur_node.left_child = self.generate_tree(leftData, leftLabel)
        cur_node.right_child = self.generate_tree(rightData, rightLabel)
        return cur_node

    def select_feature(self,data,label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        
        lowestEntropy = float("inf")
        
        for i in range(len(data[0])):    #0 ~ 63
            # compute the entropy of splitting based on the selected features
            featureData = data[:,i]     #3000 x 1 vector             
            zeros = np.asarray(np.where(featureData == 0))
            ones = np.asarray(np.where(featureData == 1))   
            #print(zeros)
            zerosLabels = label[zeros[0]]
            onesLabels = label[ones[0]]            
            entropy = self.compute_split_entropy(zerosLabels, onesLabels)            
            # select the feature with minimum entropy
            if entropy < lowestEntropy:
                lowestEntropy = entropy
                best_feat = i
            
        return best_feat

    def compute_split_entropy(self,left_y, right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two branches
        split_entropy = 0 # placeholder
        
        leftEntropy = self.compute_node_entropy(left_y)
        rightEntropy = self.compute_node_entropy(right_y)
        leftSamples = len(left_y)
        rightSamples = len(right_y)
        total = leftSamples + rightSamples
        split_entropy = leftEntropy * leftSamples/total + rightEntropy * rightSamples/total
    
        return split_entropy

    def compute_node_entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        node_entropy = 0 # placeholder
        e = 1e-15
        #print(len(label)) #3000
        totalSamples = len(label)
        
        count = np.bincount(label)
        count = count / totalSamples
        for j in range(len(count)):
            node_entropy += -count[j] * np.log2(count[j] + e)
        return node_entropy
