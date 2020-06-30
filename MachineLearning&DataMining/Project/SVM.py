
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
print(data["type"].value_counts())
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
            # change emoji to word
            new_post = new_post.replace(":)"," smile ")
            new_post = new_post.replace(":("," sad ")
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
        with open("posts.corpus","w") as corpus:
            for post in user_posts:
                corpus.write(" ".join(post) + "\n")
        print("Saved to posts.corpus!")
    else:
        user_posts = pickle.load(open(filename,"rb"))
        print("Loaded user posts")
    return user_posts

user_posts = generate_posts()


# ## Generate TF-IDF model

# In[ ]:


def generate_tfidf(user_posts, path="", retrain=False):
    filename = os.path.join(path,"tfidf.npy")
    if not os.path.isfile(filename) or retrain:
        word_lst = []
        for post in user_posts:
            word_lst += post

        # make dictionary (used for TF-IDF)
        word_counts = Counter(word_lst)
        # remove words that don't occur too frequently
        print("# of words before:",len(word_counts))
        for word in list(word_counts): # avoid changing size
            if word_counts[word] < 20:
                del word_counts[word]
        print("# of words after:",len(word_counts))

        # generate IDF value
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_word = {k: w for k, w in enumerate(sorted_vocab)}
        word_to_int = {w: k for k, w in int_to_word.items()}
        np.save("int2word.npy",int_to_word)
        idf = np.zeros((len(sorted_vocab),))
        for uid, post in enumerate(user_posts):
            set_words = set(post) # avoid duplication
            for word in set_words:
                if word in sorted_vocab:
                    idf[word_to_int[word]] += 1 # count frequency
            if uid * 100 % n_users == 0:
                print("Done {}/{} = {}%".format(uid,n_users,uid*100/n_users))
        idf = np.log(len(user_posts) / (idf + 1)) # avoid divided by 0
        print("Finished generating IDF values")
        np.save("idf.npy",idf)

        # generate TF value
        tfidf_values = np.zeros((len(user_posts),len(idf)))
        for i, post in enumerate(user_posts):
            for post_word in post:
                idx = word_to_int.get(post_word,None)
                if idx != None:
                    tfidf_values[i][idx] += 1
            if len(post) != 0:
                tfidf_values[i] /= len(post)
        print("Finished generating TF values")
        tfidf_values *= idf
        print(tfidf_values.shape)
        np.save(filename,tfidf_values)
        print("Saved to {}!".format(filename))
    else:
        tfidf_values = np.load(filename,allow_pickle=True)
        print("Loaded {}".format(filename))
    n_words = tfidf_values.shape[1]
    print('Vocabulary size:', n_words)
    return tfidf_values

tfidf = generate_tfidf(user_posts,retrain=False)


# ## Generate Word2Vec model

# In[ ]:


import multiprocessing
from gensim.models import Word2Vec
EMBEDDING_SIZE=512


# In[ ]:


def train_w2v(RETRAIN=True):
    if not os.path.isfile("posts.bin") or RETRAIN:
        print("Training Word2Vec Model ...")
        start = time.time()
        w2v = Word2Vec(corpus_file="posts.corpus",size=EMBEDDING_SIZE,
                       window=5,min_count=20,iter=30,
                       workers=multiprocessing.cpu_count())
        end = time.time()
        print("Word2Vec Time: {:.2f}s".format(end - start))
        w2v.save("posts.bin")
        w2v.wv.save_word2vec_format("posts.vec",binary=False)
    else:
        w2v = Word2Vec.load("posts.bin")
        print("Loaded pretrained Word2Vec Model")
    return w2v

wv = train_w2v()


# In[ ]:


def aggregate_wv(n_users,user_posts,w2v,filename,RETRAIN=False):
    if not os.path.isfile(filename+".npy") or RETRAIN:
        print("Generating sentence vectors ...")
        begin_time = time.time()
        user_vec = np.zeros((n_users,EMBEDDING_SIZE))
        for uid, post in enumerate(user_posts):
            cnt = 0
            for uid, word in enumerate(post):
                try:
                    user_vec[uid] += w2v.wv[word]
                    cnt += 1
                except:
                    pass
            if cnt != 0: # avoid divided by 0
                user_vec[uid] /= cnt
            if uid * 10 % n_users == 0:
                print("Done {}%={}/{}".format(uid*100//n_users, uid, n_users),flush=True)
        end_time = time.time()
        np.save(filename,user_vec)
        print("Finished! Time: {:.2f}s".format(end_time - begin_time))
    else:
        user_vec = np.load(filename)
        print("Loaded aggregated sentence vectors ({})".format(filename))
    return user_vec

X_vec = aggregate_wv(n_users,user_posts,wv,"w2v.feat")
print(X_vec.shape)


# ## Random Forest model

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


# In[ ]:


def split_balanced(data, target, test_size=0.2):
    classes = np.unique(target)
    n_test = np.round(len(target) * test_size)
    n_train = max(0, len(target) - n_test)

    idxs = []
    n_train_class = []
    n_test_class = []
    for i, cl in enumerate(classes):
        n_in_class = np.sum(target == cl)
        n_train_class.append(int(np.round(n_in_class * (1 - test_size))))
        n_test_class.append(max(0, n_in_class - n_train_class[i]))
        idxs.append(np.random.choice(np.nonzero(target == cl)[0],
                                     n_train_class[i] + n_test_class[i],
                                     replace=False))

    idx_train = np.concatenate([idxs[i][:n_train_class[i]]
                                for i in range(len(classes))])
    idx_test = np.concatenate([idxs[i][n_train_class[i]:n_train_class[i] + n_test_class[i]]
                               for i in range(len(classes))])

    X_train = data[idx_train,:]
    X_test = data[idx_test,:]
    y_train = target[idx_train]
    y_test = target[idx_test]

    return X_train, X_test, y_train, y_test


# In[ ]:


X = tfidf
# X = X_vec
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# sss = StratifiedShuffleSplit(test_size=0.3)
X_train, X_test, y_train, y_test = split_balanced(X, Y)


# In[ ]:


def random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_jobs=multiprocessing.cpu_count()) # use all processors
    clf.fit(X_train, y_train)
    predict = clf.score(X_test, y_test)
    print("Random Forest acc: {:.2f}%".format(predict * 100))
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred,target_names=labels))
    return clf


# In[ ]:


rf = random_forest(X_train,y_train,X_test,y_test)


# In[ ]:


def SVMClassifier(X_train, y_train, X_test, y_test):
    begin_time = time.time()
    clf = OneVsRestClassifier(SVC(kernel="linear"),n_jobs=multiprocessing.cpu_count())
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("Finished! Time: {:.2f}s".format(end_time - begin_time))
    y_pred = clf.predict(X_test)
    acc = np.sum(y_pred == y_test) / len(y_pred)
    print("Support Vector Machine (SVM) acc: {:.2f}%".format(acc * 100))
    print(classification_report(y_test,y_pred,target_names=labels))
    return clf


# In[ ]:


svm = SVMClassifier(X_train, y_train, X_test, y_test)


# In[ ]:


# for train_idx, test_idx in sss.split(X, Y):
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = Y[train_idx], Y[test_idx]
#     random_forest(X_train, y_train, X_test, y_test)


# In[ ]:


pickle.dump(svm,open("svm.pkl","wb"))

