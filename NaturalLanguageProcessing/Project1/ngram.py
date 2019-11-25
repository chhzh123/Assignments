import sys, os
import pickle
from collections import Counter
import jieba

N = 3
original_path = "dict_jieba"
generated_path = "dict_no_stop_jieba"
stopwords = [word[:-1] for word in open("stopwords.txt","r",encoding="utf-8")] # delete \n

word_counter_file_name = "word_counter.pkl"
ngram_counter_file_name = "ngram_counter_{}.pkl".format(N)
ngram_counter_pre_file_name = "ngram_pre_counter_{}.pkl".format(N)

if False:
    for i,file_name in enumerate(os.listdir(original_path),1):
        with open("{}/{}".format(original_path,file_name),"r",encoding="utf-8") as infile:
            new_words = []
            for j,line in enumerate(infile):
                if j == 0 or line[0] == "（":
                    continue
                new_words += ["<BOS>"]
                for word in line.split():
                    if word not in stopwords:
                        new_words.append(word)
                new_words += ["<EOS>"]
            outfile = open("{}/{}".format(generated_path,file_name),"w",encoding="utf-8")
            outfile.write(' '.join(new_words))
        if i % 100 == 0:
            print("Finish {}/{}".format(i,len(os.listdir(original_path))))
# sys.exit()

def generate_ngram(token,n,file_flag=False):
    if file_flag:
        token = open(token,"r",encoding="utf-8").read().split()
    ngrams = zip(*[token[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def cal_ngram(ngrams):
    product = 1
    for item in ngrams:
        item_pre = " ".join(item.split()[:-1])
        # additive smoothing
        product *= (ngram_counter.get(item,0) + 1) / (len(ngram_pre_counter) + ngram_pre_counter.get(item_pre,0))
    return product

if False:
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

    word_counter = Counter(word_lst)
    ngram_pre_counter = Counter(ngram_pre_lst)
    ngram_counter = Counter(ngram_lst)
    with open(word_counter_file_name,"wb") as f:
        pickle.dump(word_counter,f)
    with open(ngram_counter_file_name,"wb") as f:
        pickle.dump(ngram_counter,f)
    with open(ngram_counter_pre_file_name,"wb") as f:
        pickle.dump(ngram_pre_counter,f)

# sys.exit()
word_counter = pickle.load(open(word_counter_file_name,"rb"))
ngram_pre_counter = pickle.load(open(ngram_counter_pre_file_name,"rb"))
ngram_counter = pickle.load(open(ngram_counter_file_name,"rb"))
# word_lst = sorted(list(word_counter.items()),key=lambda x: x[1],reverse=True)
# print(word_lst[:100])
# sys.exit()

print("Finish generating word dict!")

groundtrue = [line[:-1] for line in open("answer.txt","r",encoding="utf-8")]
acc = 0

myanswer = open("myanswer.txt","w",encoding="utf-8")

print("Use {}-gram model to predict".format(N))
with open("questions.txt","r",encoding="utf-8") as question_file:
    for i,question_str in enumerate(question_file,1):
        q_index = question_str.index("[MASK]")
        question_pre, question_post = question_str[:q_index], question_str[q_index+len("[MASK]"):]
        max_val = (-0x3f3f3f3f,None)
        seg_pre, seg_post = jieba.lcut(question_pre,cut_all=False), jieba.lcut(question_post,cut_all=False)
        seg_pre.insert(0,"<BOS>")
        seg_post.insert(len(seg_post),"<EOS>")
        seg_pre_lst, seg_post_lst = [], []
        for word in seg_pre:
            if word not in stopwords and word != "\n":
                seg_pre_lst.append(word)
        for word in seg_post:
            if word not in stopwords and word != "\n":
                seg_post_lst.append(word)
        for mask in word_counter.keys():
            if mask in ["<BOS>","<EOS>"]:
                continue
            new_text = seg_pre_lst[-N+1:] + [mask] + seg_post_lst[:N-1]
            text_gram = generate_ngram(new_text,N)
            gram_val = cal_ngram(text_gram)
            if gram_val > max_val[0]:
                max_val = (gram_val,mask)
        if max_val[1] == groundtrue[i-1]:
            acc += 1
            print("{}√ [MASK] = {} ({}) - {}".format(i,max_val[1],max_val[0],groundtrue[i-1]),flush=True)
        else:
            print("{} [MASK] = {} ({}) - {}".format(i,max_val[1],max_val[0],groundtrue[i-1]),flush=True)
        myanswer.write("{}\n".format(max_val[1]))

print("Accuracy: {:.2f}%".format(acc))