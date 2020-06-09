
# coding: utf-8

# In[ ]:


import os, sys, time
import multiprocessing
import pickle
import re, string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# ## Preprocess

# In[ ]:


data = pd.read_csv("mbti_1.csv")
n_users = len(data)
posts = data["posts"]
labels = data["type"].unique()
n_class = len(labels)
type2num = {label: i for i,label in enumerate(labels)}
Y = np.array(list(map(lambda s: type2num[s], data["type"].to_numpy())))


# In[ ]:


def plot_distribution():
    fig, ax = plt.subplots(figsize=(10,4))
    type_val = data["type"].value_counts()
    labels = type_val.keys()
    x = np.arange(len(labels))
    ax.bar(x, type_val.values)
    ax.set_ylabel("# of people")
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation='45')
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    fig.tight_layout()
    plt.show()


# In[ ]:


def generate_posts(path=""):
    filename = os.path.join(path,"posts.pkl")
    user_posts = []
    if not os.path.isfile(filename):
        stopwords = pd.read_csv("stopwords.csv").to_numpy().reshape(-1)
        stopwords = np.array(list(map(lambda s: s.replace("'",""),stopwords)))
        for uid in range(n_users):
            # add empty space first (better used for regex parsing)
            new_post = posts[uid].replace("|||"," ||| ")
            new_post = new_post.replace(",",", ")
            # remove url links
            new_post = re.sub("(http|https):\/\/.*?( |'|\")","",new_post)
            # avoid words in two sentences merged together after removing spaces
            new_post = new_post.replace(".",". ")
            # remove useless numbers and punctuations
            new_post = re.sub(r"[0-9]+", "", new_post)
            new_post = new_post.translate(str.maketrans('', '', string.punctuation))
            # remove redundant empty spaces
            new_post = re.sub(" +"," ",new_post).strip()
            # make all characters lower
            new_post = new_post.lower()
            temp = []
            # remove stopping words
            for word in new_post.split():
                if len(word) != 1 and word not in stopwords:
                    temp.append(word)
            user_posts.append(temp)
            if uid * 100 % n_users == 0:
                print("Done {}/{} = {}%".format(uid,n_users,uid*100/n_users))
        print("Finished generating word list")
        pickle.dump(user_posts,open(filename,"wb"))
    else:
        user_posts = pickle.load(open(filename,"rb"))
        print("Loaded user posts")
    return user_posts


# ## Generate BoW model

# In[ ]:


def generate_dict(user_posts,path=""):
    filename = os.path.join(path,"word_dict.npz")
    if not os.path.isfile(filename):
        word_lst = []
        for post in user_posts:
            word_lst += post

        # make dictionary (used for bag of words, BOW)
        word_counts = Counter(word_lst)
        word_counts["<UNK>"] = max(word_counts.values()) + 1
        # remove words that don't occur too frequently
        print("# of words before:",len(word_counts))
        for word in list(word_counts): # avoid changing size
            if word_counts[word] < 6:
                del word_counts[word]
        print("# of words after:",len(word_counts))
        # sort based on counts, but only remain the word strings
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

        # make embedding based on the occurance frequency of the words
        int_to_word = {k: w for k, w in enumerate(sorted_vocab)}
        word_to_int = {w: k for k, w in int_to_word.items()}
        np.savez(filename,int2word=int_to_word,word2int=word_to_int)
    else:
        infile = np.load(filename,allow_pickle=True)
        int_to_word = infile["int2word"].item()
        word_to_int = infile["word2int"].item()
        print("Loaded {}".format(filename))
    n_words = len(int_to_word)
    print('Vocabulary size:', n_words)
    return word_to_int, int_to_word


# In[ ]:


def generate_bow(user_posts,word_to_int):
    filename = "bow.npy"
    if not os.path.isfile(filename):
        n_users = len(user_posts)
        n_words = len(word_to_int)
        feature = np.zeros((n_users,n_words))
        print(feature.shape)
        for uid, post in enumerate(user_posts):
            count = Counter(post)
            for key in count:
                feature[uid][word_to_int.get(key,0)] = count[key]
            if uid * 100 % n_users == 0:
                print("Done {}/{} = {}%".format(uid,n_users,uid*100/n_users))
        print("Finished generating BoW model")
        np.save(filename,feature)
        print("Saved {}".format(filename))
    else:
        feature = np.load(filename)
        print("Loaded BoW model")
    return feature


# In[ ]:


user_posts = generate_posts()
word2int, int2word = generate_dict(user_posts)
X = generate_bow(user_posts,word2int)


# ## Random Forest model

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report


# In[ ]:


N_TREES = 5
MAX_DEPTH = 10


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
unique_X_val = []
for attr in range(X.shape[1]): # disadvantage
    unique_X_val.append(np.unique(X[:,attr]))


# In[ ]:


def sklearn_rf(n_trees=5): # used for comparison
    from sklearn.ensemble import RandomForestClassifier
    print("Begin training...")
    clf = RandomForestClassifier(n_estimators=n_trees,verbose=2,n_jobs=4) # use all processors
    clf.fit(X_train, y_train)
    predict = clf.score(X_test, y_test)
    print("Random forest acc: {:.2f}%".format(predict*100))


# In[ ]:


sklearn_rf(N_TREES)


# In[ ]:


def mode(arr):
    """
    Input: 1D numpy array
    Output: mode of this array
    """
    counts = np.bincount(arr)
    return np.argmax(counts)

def value_counts(arr,normalize=False):
    """
    Input: arr (numpy array)
    Output: Counts of unique elements in arr
    """
    unique, counts = np.unique(arr, return_counts=True)
    counts = counts if not normalize else (counts / counts.sum())
    return unique, counts


# In[ ]:


def entropy(p):
    """
    Input: p (prob) is a numpy array
    Output: Ent = - \sum_i p_i \log p_i
    """
    if p.ndim == 1:
        new_p = p[p != 0]
        return -np.sum(new_p * np.log2(new_p))
    else: # high dimensional input (should be guaranteed no zeros exist)
        return -np.sum(p * np.log2(p),axis=1)
    
def information_gain(D,a,L):
    """
    Input: D (dataset), a (attribute), L (labels)
        attributes are features, and all discrete
    Output: Gain = Ent(D) - \sum_v |D_v|/|D| Ent(D_v)
        where v\in V is the unique values of a
    """
    _, pk = value_counts(L,normalize=True)
    V, prop_Dv = value_counts(D[:,a],normalize=True) # proportion = |D_v|/|D|
    # prob (|V|,|class|)
    sumup = 0
    for av, prop in zip(V,prop_Dv):
        _, prob_Dv = value_counts(L[D[:,a] == av],normalize=True)
        sumup += prop * entropy(prob_Dv)
    return (entropy(pk) - sumup, a)


# In[ ]:


class Node:
    """
    Class Node in decision tree
    """

    def __init__(self):
        self.branch = {}

    def setLeaf(self,catagory,cnt=1):
        """
        Set this node as a leaf with "catagory"
        """
        self.label = "Leaf"
        self.catagory = catagory

    def setBranch(self,attr,value,node):
        """
        Set this node as a parent node with "attr"
        Add a child "node" with "value"
        """
        self.label = "Branch"
        self.attr = attr
        self.branch[value] = node


# In[ ]:


class ID3:
    """
    ID3 Decision Tree with pre-partition
    """

    def __init__(self,train_set=None,test_set=None,
                 attributes=None,unique_val=None,
                 prune=False,tid=0,max_depth=5,random=False):
        """
        Add datasets into the class
        """
        self.train_set = train_set
        self.test_set = test_set
        self.attributes = attributes
        self.unique_val = unique_val
        self.prune = prune
        self.tid = tid
        self.max_depth = max_depth
        self.random = random
        print("Loaded dataset. # features: {}".format(len(self.attributes)))

    def TreeGenerate(self,dataset,labels,attributes,depth,
                     cnt_leaves=0,root=None,prev_best=None):
        """
        Core algorithm of ID3 that generates the whole tree
        """
        catagory = np.unique(labels)
        node = Node() if root == None else root # better used for validation indexing
        cnt_leaves += 1

        # 1) All samples in "dataset" belongs to the same catagory
        if len(catagory) == 1:
            node.setLeaf(catagory[0],cnt_leaves)
            return node

        # 2) "attributes" is empty, or the values of "dataset" on "attributes" are the same
        if len(attributes) == 0 or           np.array([len(np.unique(dataset[:,a])) == 1 for a in attributes]).all() == True:
            node.setLeaf(mode(labels),cnt_leaves)
            return node

        """The general case"""
        # find the attribute with greatest information gain
        max_gain = (-0x3f3f3f3f,None)
        if self.random:
            k = int(np.log2(len(attributes))) + 1 # random set!!!
        else:
            k = len(attributes)
        random_attributes = attributes[np.random.choice(len(attributes),k,replace=False)]
        for a in random_attributes:
            gain = information_gain(dataset,a,labels)
            if gain[0] > max_gain[0]:
                a_best, max_gain = a, gain
        unique = self.unique_val[a_best] # be careful, not dataset!
        num_leaves = len(unique)
        print("Tree {} Depth {}: {} - {}\t # leaves: {}".format(self.tid, depth, prev_best, max_gain, num_leaves))
        
        if self.prune and self.test_set != None:
            # without partition
            node.setLeaf(mode(labels),cnt_leaves)
            acc_without_partition = self.test(val=True)

            # with partition (make branches)
            for av in unique:
                Dv = dataset[dataset[:,a_best] == av,:]
                labels_v = labels[dataset[:,a_best] == av]
                cnt_leaves += 1
                leafnode = Node()
                if len(Dv) == 0:
                    leafnode.setLeaf(mode(labels),cnt_leaves)
                else:
                    leafnode.setLeaf(mode(labels_v),cnt_leaves)
                node.setBranch(a_best,av,leafnode)
            print("Depth {} ({}): ".format(depth,cnt_leaves))
            acc_with_partition = self.test(val=True,print_flag=True)

            # pre-pruning (to make sure it has generated sufficient nodes, depth is set here)
            if depth > 5 and acc_without_partition >= acc_with_partition:
                cnt_leaves -= num_leaves
                print("Prune at {}: {} (without) >= {} (with)".format(a_best,acc_without_partition,acc_with_partition))
                node.setLeaf(mode(labels))
                return node
            elif depth > 5:
                print(a_best,acc_without_partition,acc_with_partition)

            # true partition (branching makes more gains)
            for av in unique:
                Dv = dataset[dataset[:,a_best] == av,:]
                labels_v = labels[dataset[:,a_best] == av]
                # 3) "Dv" is empty, which can not be partitioned
                if len(Dv) != 0:
                    node.setBranch(a_best,av,self.TreeGenerate(Dv,labels_v,
                                                               attributes[attributes != a_best],
                                                               depth+1,cnt_leaves))
        else:
            if depth > self.max_depth:
                node.setLeaf(mode(labels),cnt_leaves)
                return node
            for av in unique:
                Dv = dataset[dataset[:,a_best] == av,:]
                labels_v = labels[dataset[:,a_best] == av]
                cnt_leaves += 1
                leafnode = Node()
                if len(Dv) == 0:
                    leafnode.setLeaf(mode(labels),cnt_leaves)
                    node.setBranch(a_best,av,leafnode)
                else:
                    node.setBranch(a_best,av,self.TreeGenerate(Dv,labels_v,
                                                               attributes[attributes != a_best],
                                                               depth+1,cnt_leaves,prev_best=a_best))
        return node

    def train(self,train_set=None):
        """
        Train the decision tree
        """
        if train_set != None:
            self.train_set = train_set
        start_time = time.time()
        self.root = Node()
        self.root = self.TreeGenerate(self.train_set[0],self.train_set[1],
                                      self.attributes,
                                      depth=1,root=self.root,
                                      cnt_leaves=0)
        print("Tree {} Time: {:.2f}s".format(self.tid, time.time() - start_time))

    def predict(self,feat):
        p = self.root
        while p.label != "Leaf": # get to the leaf node
            p = p.branch[feat[p.attr]]
        return p.catagory

    def test(self,val=False,print_flag=False):
        """
        Testing/Validation and calculate the accuracy
        """
        acc = 0
        test_X = self.test_set[0]
        test_Y = self.test_set[1]
        for i, row in enumerate(test_X):
            pred = self.predict(row)
            if pred == test_Y[i]:
                acc += 1
        acc /= len(test_X)
        if print_flag:
            print("Accurary: {:.2f}%".format(acc * 100))
        return acc


# In[ ]:


def watermelon_test(): # used for checking the correctness
    watermelon = pd.read_csv("watermelon.csv",header=None).to_numpy()
    X_train = watermelon[:,:6].astype(int)
    y_train = watermelon[:,-1].astype(int).reshape(-1)
    unique_X_val = []
    for attr in range(X_train.shape[1]): # disadvantage
        unique_X_val.append(np.unique(X_train[:,attr]))
    dt = ID3(train_set=(X_train,y_train),
             test_set=(X_train,y_train),
             attributes=np.arange(0,6),
             unique_val=unique_X_val)
    dt.train()

# watermelon_test()


# In[ ]:


class RandomForest():
    
    def __init__(self,n_trees=3,max_depth=5):
        self.n_trees = n_trees
        self.dt = [0] * self.n_trees
        self.max_depth = max_depth
        
    def fit(self,train_X,train_Y):
        start_time = time.time()
        tasks = []
        for i in range(self.n_trees):
            def train_dt(idx):
                samp_X, samp_Y = resample(train_X,train_Y,n_samples=len(train_X))
                print("Building tree {}".format(idx))
                dt = ID3(train_set=(samp_X.astype(int), samp_Y.astype(int)),
                         attributes=np.array(list(range(train_X.shape[1]))),
                         unique_val=unique_X_val,
                         prune=False,
                         max_depth=self.max_depth,
                         tid=idx,
                         random=True)
                dt.train()
                pickle.dump(dt,open("tree-{}.pkl".format(idx),"wb"))
            p = multiprocessing.Process(target=train_dt,args=(i,))
            tasks.append(p)
        [x.start() for x in tasks]
        [x.join() for x in tasks]
        for i in range(self.n_trees):
            self.dt[i] = pickle.load(open("tree-{}.pkl".format(i),"rb"))
        print("Random forest time: {}s".format(time.time() - start_time))

    def predict(self,test_X):
        print("Generating prediction...")
        pred = np.zeros((len(test_X),))
        for i, row in enumerate(test_X):
            counts = np.zeros((len(labels),))
            for j in range(self.n_trees):
                pred_one = self.dt[j].predict(row)
                counts[pred_one] += 1
            pred[i] = np.argmax(counts)
        return pred

    def score(self,test_X,test_Y):
        pred_Y = self.predict(test_X)
        acc = np.sum(pred_Y == test_Y) / len(test_Y)
        print(classification_report(pred_Y,test_Y,target_names=labels))
        return acc


# In[ ]:


rf = RandomForest(n_trees=N_TREES,max_depth=MAX_DEPTH)
rf.fit(X_train,y_train)
acc = rf.score(X_test,y_test)
print("Acc: {:.2f}%".format(acc * 100))

