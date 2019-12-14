
# coding: utf-8

# In[1]:


import logging

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log/bp-40.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# In[2]:


import os, sys, time
import pandas as pd
import numpy as np

# 0: Continuous
# 1: Nominal
# 2: Categorical / Discrete
attr_dict = {"surgery": 1,
 "Age": 2,
 "Hospital Number": 1,
 "rectal temperature": 0,
 "pulse": 0,
 "respiratory rate": 0,
 "temperature of extremities": 2,
 "peripheral pulse": 2,
 "mucous membranes": 1,
 "capillary refill time": 2,
 "pain": 1,
 "peristalsis": 2,
 "abdominal distension": 1,
 "nasogastric tube": 1,
 "nasogastric reflux": 2,
 "nasogastric reflux PH": 0,
 "rectal examination": 2,
 "abdomen": 1,
 "packed cell volume": 0,
 "total protein": 0,
 "abdominocentesis appearance": 1,
 "abdomcentesis total protein": 0,
 "outcome": 1,
 "surgical lesion": 1,
 "type of lesion 1": 1,
 "type of lesion 2": 1,
 "type of lesion 3": 1,
 "cp_data": 1} # 0: continuous, 1: discrete

train_data = pd.read_csv("horse-colic.data",names=attr_dict.keys(),index_col=False,delim_whitespace=True)
test_data = pd.read_csv("horse-colic.test",names=attr_dict.keys(),index_col=False,delim_whitespace=True)
# for a in train_data.columns.values:
#     print(train_data[a].value_counts())

def preprocessing(data):
    """
    Select some useful attributes
    """
    drop_attr = ["type of lesion 2", "type of lesion 3","Hospital Number","nasogastric reflux PH","abdomcentesis total protein"]
    # drop_attr = []
    attributes = []
    for a in data.columns.values:
        in_flag = attr_dict.get(a,None)
        if in_flag == None: # newly appended
            attributes.append(a)
        elif in_flag == 0 and a not in drop_attr: # continuous
            # data[a] = (data[a] - data[a].mean()) / data[a].std() # normalization
            attributes.append(a)
        else: # discrete, no need to append
            pass
    df = data[attributes]
    return df
    # normalized_df = (df-df.mean())/df.std() # (df.max()-df.min())
    # return normalized_df

def fill_data(data):
    """
    Fill in missing data (?)
    For continuous attributes, fill them with mean values
    For discrete attributes, fill them with the values appear most
    """
    for a in data.columns.values:
        if a in ["type of lesion 1", "Hospital Number"]: # remove
        # if a in ["Hospital Number"]: # remove
            continue
        if data[a].dtype != np.int64: # has ?
            have_data = data[data[a] != "?"][a]
            if attr_dict[a]: # discrete
                data.loc[data[a] == "?",a] = have_data.value_counts().idxmax() # view or copy? Use loc!
                if a != "outcome" and attr_dict[a] != 2:
                    # generate one-hot encoding
                    data[a] = pd.Categorical(data[a])
                    dummies = pd.get_dummies(data[a],prefix="{}_category".format(a))
                    data = pd.concat([data,dummies],axis=1)
            else: # continuous
                data.loc[data[a] == "?",a] = np.mean(have_data.astype(np.float))
        elif attr_dict[a] == 1:
            # generate one-hot encoding
            data[a] = pd.Categorical(data[a])
            dummies = pd.get_dummies(data[a],prefix="{}_category".format(a))
            data = pd.concat([data,dummies],axis=1)
    return data.astype(np.float)

# Data cleaning
data = pd.concat([train_data,test_data],axis=0)
data = fill_data(data)
label = data["outcome"].astype(np.float)
train_label, test_label = label[:len(train_data)], label[len(train_data):]
# train_label /= 3
train_label = [[1,0,0] if label == 1 else ([0,1,0] if label == 2 else [0,0,1]) for label in train_label]
# train_data, test_data = data[:len(train_data)], data[len(train_data):]
# train_data = preprocessing(train_data)
# test_data = preprocessing(test_data)
data = preprocessing(data)
train_data, test_data = data[:len(train_data)], data[len(train_data):]
print(train_data.columns)
print(len(train_data.columns))


# In[3]:


import nn
import time
import pickle

def get_batches(data,label,batch_size=1):
    num_batches = len(data) // batch_size
    for i in range(0,num_batches,batch_size):
        # yield data[i:i+batch_size].to_numpy(), label[i:i+batch_size].to_numpy().reshape(-1,1)
        yield data[i:i+batch_size].to_numpy(), np.array(label[i:i+batch_size])

def train(net,max_iter=1000,early_stop=False):
    """
    Train the network
    """
    start_time = time.time()
    loss_history, accuracy_history = [], []
    losses = []
    for i in range(max_iter):
        net.train()

        batches = get_batches(train_data,train_label,16) # generator
        for x, y in batches:
            Y_hat = net.forward(x)
            loss = net.MSE(Y_hat, y)
            # loss = net.cross_entropy(Y_hat,y).mean()
            losses.append(loss)
            # net.backward(Y_hat,y) # disable weight decay
            net.backward(Y_hat,y,0.1) # enable weight decay

        # logging
        if (i+1) % 100 == 0:
            avg_loss = np.array(losses).mean()
            loss_history.append(avg_loss)
            losses = []
            acc = test(net,test_data,test_label)
            accuracy_history.append(acc)
            # Early syop
            if early_stop and (i+1) % 1000 and len(accuracy_history) >= 5 and accuracy_history[-1] < min(accuracy_history[-10:-1]):
                # net.learning_rate *= 0.999 # learning rate decay
                logger.info("Early stop at iter {}/{} Loss: {} Accuracy: {}%".format(i+1,max_iter,avg_loss,acc*100))
                print("Early stop at iter {}/{} Loss: {} Accuracy: {}%".format(i+1,max_iter,avg_loss,acc*100))
                break
            logger.info("Iter {}/{} Loss: {} Accuracy: {}%".format(i+1,max_iter,avg_loss,acc*100))
            print("Iter {}/{} Loss: {} Accuracy: {}%".format(i+1,max_iter,avg_loss,acc*100))
            if acc == max(accuracy_history):
                with open("checkpoint/nn-best.pkl","wb") as netfile:
                    pickle.dump(net,netfile)
        if (i+1) % 1000 == 0:
            with open("checkpoint/nn-{}.pkl".format(i+1),"wb") as netfile:
                pickle.dump(net,netfile)
                
    print("Time: {}s".format(time.time() - start_time))
    return loss_history, accuracy_history


# In[4]:


def test(net,test_X,test_y,flag=True,print_flag=False):
    """
    Test the network
    """
    cnt = 0
    for j, x in test_X.iterrows():
        net.eval()
        Y_hat = net.forward(x.to_numpy().reshape(1,-1))
        # predicted = np.argmin(abs(np.array([1/3,2/3,1]) - Y_hat)) + 1
        # print(Y_hat,np.argmax(Y_hat))
        predicted = np.argmax(Y_hat) + 1
        y = test_y[j]
        if print_flag:
            print(Y_hat,predicted,y)
        if flag:
            if predicted == y:
                cnt += 1
        else:
            if [1 if t + 1 == predicted else 0 for t in range(3)] == y:
                cnt += 1
    return (cnt / len(test_X))


# In[5]:


net = nn.Network(len(train_data.columns.values),5,3,0.1)
# with open("checkpoint/nn-20000.pkl","rb") as netfile:
#     net = pickle.load(netfile)
#     net.learning_rate = 0.01
loss_history, accuracy_history = train(net,100000,True)


# In[8]:


"""
Plot the train loss curve
"""
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(loss_history,label="Loss")
ax2 = ax.twinx()
lns2 = ax2.plot(accuracy_history,"-r",label="Accuracy")

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc=0)
ax.set_xlabel("Iteration (x100)")
ax.set_ylabel("Loss")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0,1)

plt.savefig(r"fig/iteration.pdf",format="pdf",dpi=200)
plt.show()


# In[9]:


# test(net,train_data,train_label,False)
# test(net,test_data,test_label)

