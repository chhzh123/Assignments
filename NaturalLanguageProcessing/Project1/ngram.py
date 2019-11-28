import sys, os
import argparse
import pickle
from collections import Counter
import jieba

parser = argparse.ArgumentParser()
parser.add_argument("--N", help="n-gram",type=int,default=3)
parser.add_argument("--K", help="topk prediction",type=int,default=5)
args = parser.parse_args()

N = args.N
K = args.K
original_path = "dict_jieba"
generated_path = "dict_no_stop_jieba"
stopwords = [word[:-1] for word in open("stopwords.txt","r",encoding="utf-8")] # delete \n

answer_file_name = "myanswer/myanswer-{}gram-top{}.txt".format(N,K)
word_counter_file_name = "word_counter.pkl"
ngram_counter_file_name = "ngram_counter_{}.pkl".format(N)
ngram_counter_pre_file_name = "ngram_pre_counter_{}.pkl".format(N)

"""
# Stage 1
Delete stopwords and add marks before and after sentences
"""
if False: # To make it work, you need to set it to True
    for i,file_name in enumerate(os.listdir(original_path),1):
        with open("{}/{}".format(original_path,file_name),"r",encoding="utf-8") as infile:
            new_words = []
            for j,line in enumerate(infile):
                if j == 0 or line[0] == "（":
                    continue
                # add Begin of Sentence (BOS) mark
                new_words += ["<BOS>"]
                for word in line.split():
                    # delete stopwords
                    if word not in stopwords:
                        new_words.append(word)
                # add End of Sentence (EOS) mark
                new_words += ["<EOS>"]
            outfile = open("{}/{}".format(generated_path,file_name),"w",encoding="utf-8")
            outfile.write(' '.join(new_words))
        if i % 100 == 0:
            print("Finish {}/{}".format(i,len(os.listdir(original_path))))
# sys.exit()

"""
# Stage 2
Generate word dictionary and n-gram lists
"""
def generate_ngram(token,n,file_flag=False):
    if file_flag:
        token = open(token,"r",encoding="utf-8").read().split()
    ngrams = zip(*[token[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

if not os.path.isfile(ngram_counter_file_name): # To make it work, you need to set it to True
    word_lst = []
    ngram_pre_lst = []
    ngram_lst = []
    for file_name in os.listdir(generated_path):
        word_lst += generate_ngram("{}/{}".format(generated_path,file_name),1,True)
        ngram_pre_lst += generate_ngram("{}/{}".format(generated_path,file_name),N-1,True)
        ngram_lst += generate_ngram("{}/{}".format(generated_path,file_name),N,True)

    print(len(word_lst))
    print(len(ngram_pre_lst))
    print(len(ngram_lst))

    # build word/n-gram dictionary
    word_counter = Counter(word_lst)
    ngram_pre_counter = Counter(ngram_pre_lst)
    ngram_counter = Counter(ngram_lst)

    # save them to files
    with open(word_counter_file_name,"wb") as f:
        pickle.dump(word_counter,f)
    with open(ngram_counter_file_name,"wb") as f:
        pickle.dump(ngram_counter,f)
    with open(ngram_counter_pre_file_name,"wb") as f:
        pickle.dump(ngram_pre_counter,f)
# sys.exit()

# make sure you have generated these files
word_counter = pickle.load(open(word_counter_file_name,"rb"))
ngram_pre_counter = pickle.load(open(ngram_counter_pre_file_name,"rb"))
ngram_counter = pickle.load(open(ngram_counter_file_name,"rb"))
# word_lst = sorted(list(word_counter.items()),key=lambda x: x[1],reverse=True)
# print(word_lst[:100])
# sys.exit()

print("Finish generating word dict!")

"""
Stage 3
Use n-gram model to do text prediction
"""
def cal_ngram(ngrams):
    """
    Calculate the probability of the sentence consisting of `ngrams`
    based on previously computed n-gram and (n-1)-gram counts
    """
    product = 1
    for item in ngrams:
        item_pre = " ".join(item.split()[:-1])
        # additive smoothing
        product *= (ngram_counter.get(item,0) + 1) / (len(ngram_pre_counter) + ngram_pre_counter.get(item_pre,0))
    return product

groundtrue = [line[:-1] for line in open("answer.txt","r",encoding="utf-8")]
acc = 0
myanswer = open(answer_file_name,"w",encoding="utf-8")

def generate_seg_lst(string):
    """
    Generate segmentation list without stopwords
    """
    cut_lst = jieba.lcut(string,cut_all=False)
    res = []
    for word in cut_lst:
        if word not in stopwords and word != "\n":
            res.append(word)
    return res

print("Use {}-gram model to predict".format(N))
with open("questions.txt","r",encoding="utf-8") as question_file:
    for i,question_str in enumerate(question_file,1):
        # find out the blank
        q_index = question_str.index("[MASK]")
        question_pre, question_post = question_str[:q_index], question_str[q_index+len("[MASK]"):]

        # cut the sentence
        seg_pre = generate_seg_lst(question_pre)
        seg_post = generate_seg_lst(question_post)
        seg_pre.insert(0,"<BOS>")
        seg_post.insert(len(seg_post),"<EOS>")

        # calculate the maximum probability
        rank = []
        for mask in word_counter.keys():
            if mask in ["<BOS>","<EOS>"]:
                continue
            # only focus on the n-1 words before and after [MASK]
            new_text = seg_pre[-N+1:] + [mask] + seg_post[:N-1]
            text_gram = generate_ngram(new_text,N)
            gram_val = cal_ngram(text_gram)
            rank.append((gram_val,mask))

        # get the topk prediction
        rank.sort(key=lambda x: (x[0],x[1]),reverse=True)
        topk = [x[1] for x in rank[:K]]
        if groundtrue[i-1] in topk:
            acc += 1
            print("{}√ [MASK] = {} ({}) - {}".format(i,topk,rank[0][0],groundtrue[i-1]),flush=True)
        else:
            print("{} [MASK] = {} ({}) - {}".format(i,topk,rank[0][0],groundtrue[i-1]),flush=True)

        # write answers to file
        myanswer.write("{}\n".format(" ".join(topk)))

print("Accuracy: {:.2f}%".format(acc))