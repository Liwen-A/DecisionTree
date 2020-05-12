import sys
import csv
import math

class DecisionNode:
    """Class representing an internal node of a decision tree."""

    def __init__(self, test_name, test_index):
        self.test_name = test_name  # the name of the attribute to test at this node
        self.test_index = test_index  # the index of the attribute to test at this node

        self.children = {}  # dictionary mapping values of the test attribute to subtrees,
                            # where each subtree is either a DecisionNode or a LeafNode

    def classify(self, example):
        """Classify an example based on its test attribute value."""
        test_val = example[self.test_index]
        assert(test_val in self.children)
        return self.children[test_val].classify(example)

    def add_child(self, val, subtree):
        """Add a child node, which could be either a DecisionNode or a LeafNode."""
        self.children[val] = subtree

    def to_str(self, level=0):
        """Return a string representation of this (sub)tree."""
        prefix = "\t"*(level+1)
        s = prefix + "test: " + self.test_name + "\n"
        for val, subtree in sorted(self.children.items()):
            s += "{}\t{}={} ->\n".format(prefix, self.test_name, val)
            s += subtree.to_str(level+1)
        return s


class LeafNode:
    """A leaf holds only a predicted class, with no test."""

    def __init__(self, pred_class, prob):
        self.pred_class = pred_class
        self.prob = prob

    def classify(self, example):
        return self.pred_class, self.prob

    def to_str(self, level):
        """Return a string representation of this leaf."""
        prefix = "\t"*(level+1)
        return "{}predicted class: {} ({})".format(prefix, self.pred_class, self.prob)

def most(L):
    """return the most frequent element in a list L"""
    return max(set(L),key = L.count)
    

def clean(data,domains):
    """clean the data if it is congress dataset"""
    replace = []
    for j in range(len(domains)-1):
        l = []
        domains[j] = ['Yea','Nay']
        for i in range(len(data)):
            l.append(data[i][j])
        replace.append(most(l))
    
    l =[]
    for i in range(len(data)):
        domains[-1] =['Democrat','Republican']
        l.append(data[i][-1])
    replace.append(most(l))
    for i in range(len(data)):
        for j in range(len(domains)-1):
            if not data[i][j] in ['Yea','Nay']:
                data[i][j] = replace[j]
        if not data[i][-1] in domains[-1]:
            data[i][-1] = replace[i]
    return replace 


def entropy(data,domains,targetIndex):
    """helper function to calculate the entropy"""
    s = []
    for i in data:
        s.append(i[targetIndex])
    e = 0
    for a in domains[targetIndex]:
        p = s.count(a)/ len(s)
        if p != 0:
            e -=  p* math.log(p,2)
    return e
    

def partition(data, attributeIndex,domains,targetIndex):
    """helper function to partition data into subsets based on given atriibute and its domain and returns information gain"""
    h = entropy(data, domains, targetIndex)
    p = {}
    for d in domains[attributeIndex]:
        p[d] = []
    for i in data:
        p[i[attributeIndex]].append(i)   
    for v in p.values():
        h -=  len(v)/len(data) * entropy(v,domains,targetIndex)
    return p,h

def bestPartition(data,domains,targetIndex,used):
    """Get the best partition based on highest information gain"""
    l = []
    for i in range(len(domains)):
        if i != targetIndex and i not in used:
            p = partition(data,i,domains,targetIndex)
            l.append((i,p[0],p[1]))
    if len(l) != 0:
        return max(l,key = lambda i: i[2])
    else:
        None

def validSplit(p,min_examples):
    for i in p.items():
        if len(i[1])  <= min_examples:
            return False
    return True 

def buildTree(data,domains,features,targetIndex, min_examples,used):
    if entropy(data,domains,targetIndex)==0 or bestPartition(data, domains, targetIndex,used) is None:
        L = []
        for i in data:
            L.append(i[targetIndex])
        label = most(L)
        p = L.count(label) / len(L)
        return LeafNode(label,p)
    else:
        index,p,h = bestPartition(data, domains, targetIndex,used)
        used.append(index)
        while validSplit(p,min_examples) != True:
            if bestPartition(data, domains, targetIndex,used) is None:
                L = []
                for i in data:
                    L.append(i[targetIndex])
                    label = most(L)
                    p = L.count(label) / len(L)
                    return LeafNode(label,p)
            index,p,h = bestPartition(data, domains, targetIndex,used)
            used.append(index)
        n = DecisionNode(features[index],index)
        for i in p.items():
            domains = [list(set(x)) for x in zip(*i[1])]
            n.add_child(i[0], buildTree(i[1], domains, features, targetIndex, min_examples,used))
        return n
    

class DecisionTree:
    """Class representing a decision tree model."""

    def __init__(self, csv_path):
        """The constructor reads in data from a csv containing a header with feature names."""
        with open(csv_path, 'r') as infile:
            csvreader = csv.reader(infile)
            self.feature_names = next(csvreader)
            self.data = [row for row in csvreader]
            self.domains = [list(set(x)) for x in zip(*self.data)]
        self.root = None
        if 'party' in self.feature_names:
            self.replace = clean(self.data,self.domains)
            
        

    def learn(self, target_name, min_examples=0):
        """Build the decision tree based on entropy and information gain.

        Args:
            target_name: the name of the class label attribute
            min_examples: the minimum number of examples allowed in any leaf node
        """
        targetIndex = self.feature_names.index(target_name)
        self.root = buildTree(self.data,self.domains,self.feature_names,targetIndex,min_examples,[])
        #
        # TODO: Implement this method
        #
        

    def classify(self, example):
        """Perform inference on a single example.

        Args:
            example: the instance being classified

        Returns: a tuple containing a class label and a probability
        """
        if 'Yea' in example or 'Nay' in example:
            for i in range(len(example)):
                if example[i] not in ['Yea','Nay','Democrat','Republican']:
                    example[i] = self.replace[i]
        #
        # TODO: Implement this method
        #
        return self.root.classify(example)

    def __str__(self):
        return self.root.to_str() if self.root else "<empty>"




if __name__ == '__main__':

    path_to_csv = sys.argv[1]
    class_attr_name = sys.argv[2]
    min_examples = int(sys.argv[3])

    model = DecisionTree(path_to_csv)
    model.learn(class_attr_name, min_examples)
    print(model)
