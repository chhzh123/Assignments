# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("DT-prepruning-8.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("No 'fnlwgt', depth 5, no shuffle, 0.9 train data")


# %%
import pandas as pd
import numpy as np

attr_dict = {"age":0, "workclass":1, "fnlwgt":0, "education":1, "education-num":0, "marital-status":1, "occupation":1, "relationship":1, "race":1, "sex":1, "capital-gain":0, "capital-loss":0, "hours-per-week":0, "native-country":1, "salary":0} # 0: continuous, 1: discrete

train_data = pd.read_csv("adult.data",names=attr_dict.keys(),index_col=False)
test_data = pd.read_csv("adult.test",names=attr_dict.keys(),index_col=False,header=0)

def preprocessing(data):
    """
    Select some useful attributes
    """
    # attributes = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country","salary"] # discrete
    attributes = list(attr_dict.keys())
    # Since `fnlwgt` is tightly connected with `race`, `age`, `sex`, etc, which does not provide any more information, thus I simply remove the attribute here (also in order to shorten the training time).
    attributes.remove("fnlwgt")
    return data[attributes]

def fill_data(data):
    """
    Fill in missing data (?)
    """
    for a in attr_dict:
        if attr_dict[a]: # discrete
            data.loc[data[a] == " ?",a] = data[a].value_counts().argmax() # view or copy? Use loc!
        else: # continuous
            pass

# Data cleaning
train_data = preprocessing(train_data)
test_data = preprocessing(test_data)
fill_data(train_data)
fill_data(test_data)

# Generate validation set (for pre-pruning)
# train_data = train_data.sample(frac=1).reset_index(drop=True) # shuffle the data
cut = int(0.9 * len(train_data))
# cut = int(len(train_data))
train_data, validation_data = train_data[:cut], train_data[cut:]


# %%
def entropy(p):
    """
    Input: p is a numpy array
    Output: Ent = - \sum_i p_i \log p_i
    """
    if p.ndim == 1:
        new_p = p[p != 0]
        return -np.sum(new_p * np.log2(new_p))
    else: # high dimensional input (should be guaranteed no zeros exist)
        # new_p = p[(p[:,0] != 0) & (p[:,1] != 0)]
        return -np.sum(p * np.log2(p),axis=1)

def information_gain(D,a,discrete_flag=False):
    """
    Input: D (dataset), a (attribute), discrete_flag (whether a is discrete)
    Output: Gain = Ent(D) - \sum_v |D_v|/|D| Ent(D_v)
    """
    pk = D["salary"].value_counts(normalize=True).values
    if discrete_flag: # discrete
        prop_Dv = D[a].value_counts(normalize=True).values # proportion = |D_v|/|D|
        prob_Dv = np.array([D.loc[D[a] == av]["salary"].value_counts(normalize=True).get(" >50K",0) for av in D[a].unique()])
        # delete all the zero terms
        pp_stack = np.column_stack((prop_Dv,prob_Dv))
        pp_stack = pp_stack[(pp_stack[:,1] != 0) & (pp_stack[:,1] != 1)]
        prop_Dv = pp_stack[:,0]
        prob_Dv = pp_stack[:,1]
        # since it only has two categories, the probability of the other can be easily calculated
        prob_Dv_neg = 1 - prob_Dv
        return (entropy(pk) - np.sum(prop_Dv * entropy(np.column_stack((prob_Dv,prob_Dv_neg)))), a)
    else: # continuous
        # firstly sort all the existed values
        a_sort = sorted(D[a].unique())
        # calculate the partition point (a_i + a_{i+1}) / 2
        Ta = [(a_sort[i] + a_sort[i+1]) / 2 for i in range(len(a_sort)-1)]
        # find the one with minimum weighted entropy sum (\sum |D_t^\lambda|/|D| Ent(D_t^\lambda))
        min_ent, min_t = 0x3f3f3f3f, a_sort[0]
        for t in Ta: # bi-partition
            prop_Dv = len(D[D[a] < t]) / len(D)
            prop_Dv = np.array([prop_Dv,1-prop_Dv]) # proportion
            prob_Dv_smaller = D[D[a] < t]["salary"].value_counts(normalize=True).get(" >50K",0)
            prob_Dv_bigger = D[D[a] >= t]["salary"].value_counts(normalize=True).get(" >50K",0)
            prob_Dv = np.array([[prob_Dv_smaller,1-prob_Dv_smaller],[prob_Dv_bigger,1-prob_Dv_bigger]])
            # remove all zero terms
            prob_Dv = prob_Dv[(prob_Dv[:,0] != 0) & (prob_Dv[:,1] != 0)]
            # calculate weighted entropy sum
            if len(prob_Dv) == 0:
                sumup = 0
            else:
                sumup = np.sum(prop_Dv * entropy(prob_Dv))
            if min_ent > sumup:
                min_ent = sumup
                min_t = t
        # return Gain and the partition point
        return (entropy(pk) - min_ent, min_t)


# %%
class Node:
    """
    Class Node in decision tree
    """

    def __init__(self):
        self.branch = {}

    def setLeaf(self,catagory,cnt=1):
        """
        Set this node as a leaf with `catagory`
        """
        logger.info("{} - Create leaf: {}".format(cnt,catagory))
        if cnt % 10 == 0:
            print("{} - Create leaf: {}".format(cnt,catagory),flush=True)
        self.label = "Leaf"
        self.catagory = catagory

    def setBranch(self,attr,value,node,branch_value=None):
        """
        Set this node as a parent node with `attr`
        Add a child `node` with `value`
        If the attribute is continuous, `branch_value` is also given
        """
        logger.info("Create branch: {} ({})".format(attr,value))
        self.label = "Branch"
        self.attr = attr
        self.branch[value] = node
        if branch_value != None:
            self.branch_value = branch_value


# %%
import time,sys

class ID3:
    """
    ID3 Decision Tree with pre-partition and support continuous attributes
    """

    def __init__(self,train_set=None,validation_set=None,test_set=None,attr_dict=None):
        """
        Add datasets into the class
        `attr_dict` gives whether an attribute is discrete
        """
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.attr_dict = attr_dict

    def TreeGenerate(self,dataset,attributes,depth,cnt_leaves=0,root=None):
        """
        Core algorithm of ID3 that generates the whole tree
        """
        catagory = dataset["salary"].unique()
        node = Node() if root == None else root # better used for validation indexing
        cnt_leaves += 1

        # 1) All samples in `dataset` belongs to the same catagory
        if len(catagory) == 1:
            node.setLeaf(catagory[0],cnt_leaves)
            return node

        # 2) `attributes` is empty, or the values of `dataset` on `attributes` are the same
        if len(attributes) == 0 or np.array([len(dataset[a].unique()) == 1 for a in attributes]).all() == True:
            node.setLeaf(dataset["salary"].value_counts().argmax(),cnt_leaves)
            return node

        """The general case"""
        # without partition
        node.setLeaf(dataset["salary"].value_counts().argmax(),cnt_leaves)
        acc_without_partition = self.validation()

        # with partition
        # find the attribute with greatest information gain
        max_gain = (-0x3f3f3f3f,None)
        for a in attributes:
            gain = information_gain(dataset,a,self.attr_dict[a])
            if gain[0] > max_gain[0]:
                a_best, max_gain = a, gain
        num_leaves = 0
        # make branches
        if self.attr_dict[a_best]: # discrete
            num_leaves = len(self.train_set[a_best].unique())
            for av in self.train_set[a_best].unique(): # be careful, not dataset!
                Dv = dataset[dataset[a_best] == av]
                cnt_leaves += 1
                leafnode = Node()
                if len(Dv) == 0:
                    leafnode.setLeaf(dataset["salary"].value_counts().argmax(),cnt_leaves)
                else:
                    leafnode.setLeaf(Dv["salary"].value_counts().argmax(),cnt_leaves)
                node.setBranch(a_best,av,leafnode)
        else: # continuous
            num_leaves = 2
            for flag in ["Smaller","Bigger"]:
                Dv = dataset[dataset[a_best] < max_gain[1]] if flag == "Smaller" else dataset[dataset[a_best] >= max_gain[1]]
                cnt_leaves += 1
                leafnode = Node()
                if len(Dv) == 0:
                    leafnode.setLeaf(dataset["salary"].value_counts().argmax(),cnt_leaves)
                else:
                    leafnode.setLeaf(Dv["salary"].value_counts().argmax(),cnt_leaves)
                node.setBranch(a_best,flag,leafnode,branch_value=max_gain[1])
        acc_with_partition = self.validation()

        # pre-pruning (to make sure it has generated sufficient nodes, depth is set here)
        if depth > 5 and acc_without_partition >= acc_with_partition:
            cnt_leaves -= num_leaves
            print("Prune at {}: {} (without) >= {} (with)".format(a_best,acc_without_partition,acc_with_partition))
            logger.info("Prune at {}: {} (without) >= {} (with)".format(a_best,acc_without_partition,acc_with_partition))
            node.setLeaf(dataset["salary"].value_counts().argmax())
            return node
        elif depth > 5:
            print(a_best,acc_without_partition,acc_with_partition)

        # true partition (branching makes more gains)
        if self.attr_dict[a_best]: # discrete
            for av in self.train_set[a_best].unique(): # be careful, not dataset!
                Dv = dataset[dataset[a_best] == av]
                # 3) `Dv` is empty, which can not be partitioned
                if len(Dv) != 0:
                    node.setBranch(a_best,av,self.TreeGenerate(Dv,attributes[attributes != a_best],depth+1,cnt_leaves))
        else: # continuous
            for flag in ["Smaller","Bigger"]:
                Dv = dataset[dataset[a_best] < max_gain[1]] if flag == "Smaller" else dataset[dataset[a_best] >= max_gain[1]]
                if len(Dv) != 0:
                    node.setBranch(a_best,flag,self.TreeGenerate(Dv,attributes,depth+1,cnt_leaves),branch_value=max_gain[1])
        return node

    def train(self,train_set=None):
        """
        Train the decision tree
        """
        if train_set != None:
            self.train_set = train_set
        start_time = time.time()
        self.root = Node()
        self.root = self.TreeGenerate(self.train_set,self.train_set.columns.values[self.train_set.columns.values != "salary"],depth=1,root=self.root,cnt_leaves=0)
        logger.info("Time: {:.2f}s".format(time.time()-start_time))
        print("Time: {:.2f}s".format(time.time()-start_time))

    def validation(self,validation_set=None):
        """
        Validate the partition on validation set
        """
        if validation_set != None:
            self.validation_set = validation_set
        acc = 0
        for i,row in self.validation_set.iterrows():
            p = self.root
            while p.label != "Leaf": # get to the leaf node
                if self.attr_dict[p.attr]: # discrete
                    p = p.branch[row[p.attr]]
                else: # continuous
                    p = p.branch["Smaller"] if row[p.attr] < p.branch_value else p.branch["Bigger"]
            if p.catagory == row["salary"]:
                acc += 1
        acc /= len(self.validation_set)
        return acc

    def test(self,test_set=None):
        """
        Final tesing and calculate the accuracy
        """
        if test_set != None:
            self.test_set = test_set
        acc = 0
        for i,row in self.test_set.iterrows():
            p = self.root
            while p.label != "Leaf": # get to the leaf node
                if self.attr_dict[p.attr]: # discrete
                    p = p.branch[row[p.attr]]
                else: # continuous
                    p = p.branch["Smaller"] if row[p.attr] < p.branch_value else p.branch["Bigger"]
            if p.catagory == row["salary"][:-1]: # be careful of "."
                acc += 1
        acc /= len(self.test_set)
        logger.info("Accurary: {:.2f}%".format(acc * 100))
        print("Accurary: {:.2f}%".format(acc * 100))
        return acc


# %%
dt = ID3(train_set=train_data,validation_set=validation_data,test_set=test_data,attr_dict=attr_dict)
dt.train()
dt.test()

