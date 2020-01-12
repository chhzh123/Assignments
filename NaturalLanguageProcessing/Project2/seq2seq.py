
# coding: utf-8

# ## Seq2seq Model with Attention for Chinese-English Machine Translation
# 
# Some references on seq2seq:
# * Pytorch, *seq2seq translation tutorial*, <https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>
# * Practical Pytorch, *Batched seq2seq*, <https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb>
# 
# ![Seq2seq Model](https://pytorch.org/tutorials/_images/seq2seq.png)

# Some tricky things:
# * Three types of dashes in English:
#     * The Hypen (-)
#     * The En-dash (–)
#     * The Em-dash (—)
#     * Please refer to [Wikipedia]() or [English Language Help Desk](http://site.uit.no/english/punctuation/hyphen/) for more details

# In[ ]:


import re
import os
import sys
import time
import random
import logging, pseudologger
import pickle
import jieba
from collections import Counter
from argparse import Namespace

flags = Namespace(
    checkpoint_path='checkpoint',
    log_flag=True,
    log_path="log",
    data_path="data",
    seq_size=32,
    batch_size=32,
    embedding_size=256, # embedding dimension
    lstm_size=256, # hidden dimension
    gradients_norm=5, # gradient clipping
    trunc=-1,
    top_k=5,
    num_epochs=30,
    learning_rate=0.01,
    learning_rate_decay_epochs=10,
    learning_rate_decay_ratio=0.5,
    teacher_forcing_ratio=1,
    print_every=20,
    save_every=500
)

for path in [flags.checkpoint_path,flags.log_path,flags.data_path]:
    if not os.path.exists(path):
        os.mkdir(path)

if flags.log_flag:
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("{}/seq2seq-15.log".format(flags.log_path))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
else:
    logger = pseudologger.PseudoLogger()

logger.info(str(flags))


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


# In[ ]:


from nltk.stem import WordNetLemmatizer

PAD_token = 0
BOS_token = 1
EOS_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {}
        self.tmp_word_lst = []
        self.n_sentences = 0

    @staticmethod
    def normalizeString(s,lang):
        if lang == "zh":
            s = re.sub(r"&#[0-9]+;",r"",s) # so dirty!
            s = re.sub(r"�",r"",s)
            # Test if is Chinese
            # https://cloud.tencent.com/developer/article/1499958
            punc_pair = [("。","."),("！","!"),("？","?"),("，",",")]
            for zh_punc,en_punc in punc_pair:
                s = s.replace(zh_punc,en_punc)
            s = re.sub(u"[^a-zA-Z0-9\u4e00-\u9fa5,.!?]",u" ",s)
            s = re.sub(r"\s+", r" ", s)
            s = s.lower().strip()
            s = re.sub(r"[0-9]+", r" <NUM> ", s)
        else: # lang == "en"
            lemmatizer = WordNetLemmatizer() 
            s = re.sub(r"&#[0-9]+;",r"",s)
            s = re.sub(r"([,.!?])",r" \1",s) # add a space between these punctuations
            s = re.sub(r"[^a-zA-Z0-9,.!?]+",r" ",s) # remove most of the punctuations
            s = re.sub(r"\s+", r" ", s)
            s = s.lower().strip()
            s = re.sub(r"[0-9]+", r" <NUM> ", s) # replace numbers
            # lemmatization
            # https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
            # https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
            split_lst = s.split()
            split_lst = [lemmatizer.lemmatize(word) if word != "<NUM>" else "<NUM>" for word in split_lst]
            s = " ".join(split_lst)
        return s

    def addSentence(self,sentence):
        self.n_sentences += 1
        if self.name == "zh": # need to use tools to split words
            split_sentence = sentence.split(" <NUM> ")
            cut_lst = []
            for i,item in enumerate(split_sentence):
                if i != 0:
                    cut_lst += ["<NUM>"]
                cut_lst += jieba.lcut(item,cut_all=False) # precisely cut
            try: # throw out error
                if cut_lst.index("NUM"):
                    print(cut_lst)
            except:
                pass
            self.tmp_word_lst += filter(" ".__ne__,cut_lst) # remove all the white spaces
        else: # self.name == "en"
            split_lst = sentence.split()
            self.tmp_word_lst += filter(" ".__ne__,split_lst) # remove all the white spaces

    def getSentenceIndex(self,sentence,max_len,pad=True):
        """
        Do after processIndex
        """
        if self.name == "zh":
            split_sentence = sentence.split(" <NUM> ")
            cut_lst = []
            for i,item in enumerate(split_sentence):
                if i != 0:
                    cut_lst += ["<NUM>"]
                cut_lst += jieba.lcut(item,cut_all=False) # precisely cut
            filter_lst = filter(" ".__ne__,cut_lst)
            res_lst = [self.word2index.get(word,self.word2index["<PAD>"]) for word in filter_lst] + [self.word2index["<EOS>"]]
            return self.padIndex(res_lst,max_len) if pad else res_lst
        else: # self.name == "en"
            split_lst = sentence.split()
            res_lst = [self.word2index.get(word,self.word2index["<PAD>"]) for word in split_lst] + [self.word2index["<EOS>"]]
            return self.padIndex(res_lst,max_len) if pad else res_lst

    def padIndex(self,lst,max_len):
        """
        Do after processIndex
        """
        if len(lst) > max_len:
            return []
        lst += [self.word2index["<PAD>"] for i in range(max_len - len(lst))]
        return lst

    def getSentenceFromIndex(self,index_lst):
        """
        Call after processIndex
        """
        if self.name == "zh":
            return "".join([self.index2word[index] for index in index_lst])
        else:
            return " ".join([self.index2word[index] for index in index_lst])

    def processIndex(self):
        """
        Do after all the addSentence
        """
        self.word2count = Counter(self.tmp_word_lst) # {word: count}
        del self.tmp_word_lst # delete temporary word list
        max_count = max(self.word2count.values())
        self.word2count["<PAD>"] = max_count + 3 # add padding mark, label as 0
        self.word2count["<BOS>"] = max_count + 2 # add begin of sentence (BOS) mark
        self.word2count["<EOS>"] = max_count + 1 # add end of sentence (EOS) mark
        # sort based on counts, but only remain the word strings
        sorted_vocab = sorted(self.word2count, key=self.word2count.get, reverse=True)

        # make embedding based on the occurance frequency of the words
        self.index2word = {k: w for k, w in enumerate(sorted_vocab)}
        self.word2index = {w: k for k, w in self.index2word.items()}
        self.n_words = len(self.index2word)
        print('Vocabulary size of {}'.format(self.name), self.n_words)
        print(list(self.index2word.items())[:10])


# In[ ]:


def preprocess(mode="train",size=10000):
    """
    Source file in Chinese, target file in English

    Eg:
    巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。
    PARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.
    """
    data_path = flags.data_path
    zh_lang_file = "{}/zh-lang-{}-{}.pkl".format(data_path,mode,size)
    en_lang_file = "{}/en-lang-{}-{}.pkl".format(data_path,mode,size)
    pairs_file = "{}/pairs-{}-{}.pkl".format(data_path,mode,size)
    if mode == "train" and os.path.isfile(zh_lang_file) and os.path.isfile(en_lang_file) and os.path.isfile(pairs_file):
        src_lang = pickle.load(open(zh_lang_file,"rb"))
        dst_lang = pickle.load(open(en_lang_file,"rb"))
        pairs = pickle.load(open(pairs_file,"rb"))
        print('Vocabulary size of {}'.format(src_lang.name), src_lang.n_words)
        print(list(src_lang.index2word.items())[:10])
        print('Vocabulary size of {}'.format(dst_lang.name), dst_lang.n_words)
        print(list(dst_lang.index2word.items())[:10])
        return src_lang, dst_lang, pairs
    else:
        src_lang = Lang("zh")
        dst_lang = Lang("en")
        pairs = []
    path = "dataset_{}".format(size)
    set_size = 8000 if mode == "train" else 1000
    set_size = set_size * 10 if size == 100000 else set_size
    src_file = open("{}/{}_source_{}.txt".format(path,mode,set_size),"r",encoding="utf-8")
    dst_file = open("{}/{}_target_{}.txt".format(path,mode,set_size),"r",encoding="utf-8")

    print("Reading data...")
    for i,(src_line,dst_line) in enumerate(zip(src_file,dst_file),1):
        src = src_line.splitlines()[0]
        dst = dst_line.splitlines()[0]
        norm_src = Lang.normalizeString(src,"zh")
        norm_dst = Lang.normalizeString(dst,"en")
        if mode == "train":
            src_lang.addSentence(norm_src)
            dst_lang.addSentence(norm_dst)
        if i % 1000 == 0:
            print("Done {}/{}".format(i,set_size))
        pairs.append([norm_src,norm_dst])

    if mode != "train":
        return src_lang, dst_lang, pairs

    src_lang.processIndex()
    dst_lang.processIndex()

    pickle.dump(src_lang,open(zh_lang_file,"wb"))
    pickle.dump(dst_lang,open(en_lang_file,"wb"))
    pickle.dump(pairs,open(pairs_file,"wb"))
    print("Dumped to file!")
    return src_lang, dst_lang, pairs


# In[ ]:


from torch.utils import data

class TextDataset(data.Dataset):
    """
    My own text dataset
    ref: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """
    def __init__(self,mode="train",dataset_size=10000,max_seq_len=32,batch_size=32,trunc=-1):
        self.src_lang, self.dst_lang, self.pairs = preprocess(mode,dataset_size)
        print("Read {} sentence pairs".format(len(self.pairs)))
        # need to pad the sentences for easy generating Dataloader
        self.index_pairs = []
        all_pairs = []
        src_len = []
        dst_len = []
        for src, dst in self.pairs:
            src_index = self.src_lang.getSentenceIndex(src,max_seq_len,False)
            dst_index = self.dst_lang.getSentenceIndex(dst,max_seq_len,False)
            all_pairs.append([src_index,dst_index])
            src_len.append(len(src_index))
            dst_len.append(len(dst_index))
        src_len = np.array(src_len)
        dst_len = np.array(dst_len)
        print("Src avg sentence len: {}, max: {}, min: {}".format(src_len.mean(),src_len.max(),src_len.min()))
        print("Dst avg sentence len: {}, max: {}, min: {}".format(dst_len.mean(),dst_len.max(),dst_len.min()))
        self.src_len = []
        self.dst_len = []
        for src_index, dst_index in all_pairs:
            if len(src_index) <= max_seq_len and len(dst_index) <= max_seq_len:
                self.src_len.append(len(src_index))
                self.dst_len.append(len(dst_index))
                src_index = self.src_lang.padIndex(src_index,max_seq_len)
                dst_index = self.dst_lang.padIndex(dst_index,max_seq_len)
                self.index_pairs.append([src_index,dst_index])
        print("Further trimmed to {} pairs".format(len(self.index_pairs)))
        self.in_text = np.array(self.index_pairs)[:,0].reshape(-1,max_seq_len)
        self.out_text = np.array(self.index_pairs)[:,1].reshape(-1,max_seq_len)
        if trunc == -1:
            num_pairs = len(self.in_text) // batch_size * batch_size
        else:
            num_pairs = trunc
        self.in_text = self.in_text[:num_pairs]
        self.out_text = self.out_text[:num_pairs]
        self.pairs = self.pairs[:num_pairs]
        print("Use {} pairs to {}".format(num_pairs,mode))
        print("In_text shape: {}\t Out_text shape: {}".format(self.in_text.shape,self.out_text.shape))
        print("Done generating {}_{} dataset!".format(mode,dataset_size))

    def filter_pairs(self):
        self.MIN_LENGTH = {"zh":1,"en":3}
        self.MAX_LENGTH = {"zh":60,"en":150}
        filter_pairs = []
        for pair in self.pairs:
            if self.MIN_LENGTH["zh"] <= len(pair[0]) <= self.MAX_LENGTH["zh"]                 and self.MIN_LENGTH["en"] <= len(pair[1]) <= self.MAX_LENGTH["en"]:
                    filter_pairs.append(pair)
        return filter_pairs

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.in_text)

    def __getitem__(self, idx):
        """
        Generate one sample of the data
        """
        x = self.in_text[idx]
        y = self.out_text[idx]
        x_len = self.src_len[idx]
        y_len = self.dst_len[idx]
        return x, y, x_len, y_len


# ## RNN (LSTM / GRU)
# * Reference
#     * Animated RNN (LSTM & GRU), <https://towardsdatascience.com/animated-rnn-lstm-and-gru-ef124d06cf45>
#     * Pytorch LSTM, <https://pytorch.org/docs/stable/nn.html#lstm>
# 
# ![RNN](https://miro.medium.com/max/1516/1*yBXV9o5q7L_CvY7quJt3WQ.png)

# ## Encoder
# 
# ![Encoder network](https://pytorch.org/tutorials/_images/encoder-network.png)

# In[ ]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        # embed = nn.Embedding(vocab_size, vector_size)
        # "vocab_size" is the number of words in your train, val and test set
        # "vector_size" is the dimension of the word vectors you are using
        # you can view it as a linear transformation
        # the tensor is initialized randomly
        # Input: (*), LongTensor of arbitrary shape containing the indices to extract (i.e. batch size)
        # Output: (*, H), where * is the input shape and H = embedding_dim
        self.embedding = nn.Embedding(input_size, hidden_size)
        # make the embedding size equal to the hidden dimension (lstm size)
        # batch_first makes it to (batch_size, seq_len, features)
        self.lstm = nn.LSTM(hidden_size, hidden_size, dropout=self.dropout, batch_first=True)

    def forward(self, x, prev_state, input_lengths):
        """
        x: (batch_size, seq_len)
            seq_len can be viewed as the time step (many small chunks)
        embedding: (batch_size, seq_len, embedding_size)
            since batch_first flag is set to True, the first dimension is batch_size
        output: (batch_size, seq_len, embedding_size)
        h_t: (1, batch_size, hidden_size) # Actually, 1 = num_layers*num_directions
        c_t: (1, batch_size, hidden_size)

        Pytorch's pack_padded_sequence can be used to
        tackle the problem of variable length sequences
        Packs a Tensor containing padded sequences of variable length.
        
        torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
        input can be of size T x B x * where T is the length of the longest sequence (equal to lengths[0]),
        B is the batch size, and * is any number of dimensions (including 0).
        If batch_first is True, B x T x * input is expected.
        
        Reference:
        * https://discuss.pytorch.org/t/understanding-lstm-input/31110/3
        * https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.PackedSequence
        * https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        * https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        * https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
        """
        embedding = self.embedding(x)
        total_length = x.size(1) # max sequence length
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedding, input_lengths, batch_first=True) # reduce computation
        output, state = self.lstm(packed, prev_state)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=total_length) # unpack (back to padded)
        return output, state

    def forward_without_padding(self, x, prev_state):
        embedding = self.embedding(x)
        output, state = self.lstm(embedding, prev_state)
        return output, state

    def initHidden(self,batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device), # h_t
                torch.zeros(1, batch_size, self.hidden_size, device=device)) # c_t


# ## Decoder
# ![Decoder network](https://pytorch.org/tutorials/_images/decoder-network.png)

# In[ ]:


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.name = "Toy"
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1) # since CrossEntropyLoss is used

    def forward(self, x, prev_state):
        """
        x: (batch_size, seq_len)
        embedding: (batch_size, seq_len, embedding_size) # embedding_size = hidden_size
            operate the words in embedding space
        output: (batch_size, seq_len, hidden_size)
        output: (batch_size, seq_len, output_size)
            from embedding space to index space
        h_t: (1, batch_size, hidden_size)
        c_t: (1, batch_size, hidden_size)
        """
        # outputs of the encoder are passed from hidden_state
        embedding = self.embedding(x)
        embedding = F.relu(embedding)
        output, state = self.lstm(embedding, prev_state)
        output = self.linear(output)
#         output = self.softmax(output)
        return output, state

    def initHidden(self,batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device), # h_t
                torch.zeros(1, batch_size, self.hidden_size, device=device)) # c_t


# ## Decoder with Attention
# 
# Reference:
# * <https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05>
# * <https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf>
# 
# ![Decoder with Attention](https://i.imgur.com/1152PYf.png)

# In[ ]:


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=32):
        super(AttnDecoderRNN, self).__init__()
        self.name = "Attn"
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, x, prev_hidden, encoder_outputs):
        """
        x: (batch_size=32, seq_len=1, output_size)
        prev_hidden: (1, batch_size=32, hidden_size)
        encoder_outputs: (batch_size=32, seq_len=32, hidden_size) # encoder hidden states
        
        embedded: (batch_size=32, seq_len=1, hidden_size)
        decoder_output: (batch_size, seq_len=1, hidden_size)
        attn_score: dot(encoder_outputs,decoder_output)
            (batch_size,seq_len,1)
        attn_weights: softmax(attn_score)
            (batch_size,seq_len,1) -> (batch_size,1,seq_len)
        attn_output: dot(attn_weights,encoder_outputs)
            (batch_size,1,hidden_size)
        cat: cat(attn_output,decoder_output)
            (batch_size,1,2*hidden_size)
        output: (batch_size,1,output_size)
        """
        embedded = self.embedding(x)
        decoder_output, hidden = self.lstm(embedded, prev_hidden)
        attn_score = torch.bmm(encoder_outputs,decoder_output.transpose(1,2))
        attn_weights = F.softmax(attn_score,dim=1)
        attn_output = torch.bmm(attn_weights.transpose(1,2),encoder_outputs)
        cat = torch.cat((attn_output,decoder_output),dim=2)
        output = self.linear(cat)

        return output, hidden, attn_weights

    def initHidden(self,batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device), # h_t
                torch.zeros(1, batch_size, self.hidden_size, device=device)) # c_t


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plt_loss(losses):
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.savefig(r"fig/train_loss.pdf",format="pdf",dpi=200)

def train(encoder,decoder,dataset,batch_size=32,num_epochs=100,print_every=50,save_every=100):
    """
    Core training function
    """

    train_set = dataset["train_set"]
    dev_set = dataset["dev_set"]
    train_loader = data.DataLoader(dataset=train_set,batch_size=flags.batch_size,shuffle=True)
    
    # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token) # ignore padding
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=flags.learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=flags.learning_rate)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer,step_size=flags.learning_rate_decay_epochs,gamma=flags.learning_rate_decay_ratio)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer,step_size=flags.learning_rate_decay_epochs,gamma=flags.learning_rate_decay_ratio)

    iteration = 0
    losses = []

    print("Begin training ...")
    start_time = time.time()
    for e in range(num_epochs):
        encoder_ht, encoder_ct = encoder.initHidden(batch_size)
        decoder_ht, decoder_ct = decoder.initHidden(batch_size)

        for step, (x, y, x_len, y_len) in enumerate(train_loader):
            iteration += 1
            encoder.train()
            decoder.train()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            seq_lengths, perm_idx = torch.tensor(x_len).sort(0,descending=True)

            x = torch.tensor(x).to(torch.int64).to(device) # (batch_size, seq_size)
            y = torch.tensor(y).to(torch.int64).to(device) # (batch_size, seq_size)
            x = x[perm_idx]
            y = y[perm_idx]

            encoder_outputs, (encoder_ht, encoder_ct) = encoder(x, (encoder_ht, encoder_ct), seq_lengths)

            decoder_input = torch.tensor([BOS_token] * batch_size).reshape(batch_size,1).to(device) # <BOS> token
            decoder_ht, decoder_ct = encoder_ht, encoder_ct # use last hidden state from encoder
            # print(decoder_input.shape,decoder_ht.shape,decoder_ct.shape)

            # run through decoder one time step at a time
            max_dst_len = y.shape[1]
            all_decoder_outputs = torch.zeros((max_dst_len,flags.batch_size,decoder.output_size))
            if random.random() < flags.teacher_forcing_ratio:
                for t in range(max_dst_len): # for each time step
                    if decoder.name == "Toy":
                        # decoder_output: (batch_size, seq_len, output_size)
                        decoder_output, (decoder_ht, decoder_ct) = decoder(decoder_input, (decoder_ht, decoder_ct))
                        all_decoder_outputs[t] = decoder_output.transpose(1,0)
                    elif decoder.name == "Attn":
                        decoder_output, (decoder_ht, decoder_ct), decoder_attn = decoder(decoder_input, (decoder_ht, decoder_ct), encoder_outputs)
                        all_decoder_outputs[t] = decoder_output.transpose(1,0)
                    else:
                        decoder_output, (decoder_ht, decoder_ct), decoder_attn = decoder(decoder_input, (decoder_ht, decoder_ct), encoder_outputs)
                        all_decoder_outputs[t] = decoder_output
                    # teaching forcing: next input is the current target
                    decoder_input = y[:,t].reshape(batch_size,1) # remember to reshape
            else: # without teacher forcing
                for t in range(max_dst_len): # for each time step
                    if decoder.name == "Toy":
                        # decoder_output: (batch_size, seq_len, output_size)
                        decoder_output, (decoder_ht, decoder_ct) = decoder(decoder_input, (decoder_ht, decoder_ct))
                        all_decoder_outputs[t] = decoder_output.transpose(1,0)
                    elif decoder.name == "Attn":
                        decoder_output, (decoder_ht, decoder_ct), decoder_attn = decoder(decoder_input, (decoder_ht, decoder_ct), encoder_outputs)
                        all_decoder_outputs[t] = decoder_output.transpose(1,0)
                    else:
                        decoder_output, (decoder_ht, decoder_ct), decoder_attn = decoder(decoder_input, (decoder_ht, decoder_ct), encoder_outputs)
                        all_decoder_outputs[t] = decoder_output
                    # use the current output as the next input
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach().reshape(batch_size,1)

            # loss calculation
            # (max_dst_len, batch_size, output_size)
            # (batch_size, max_dst_len, output_size)
            # (batch_size, output_size, max_dst_len)
            loss = criterion(all_decoder_outputs.permute(1,2,0).to(device).to(device), y) # transpose(1,0).transpose(1,2)

            loss_value = loss.item()

            loss.backward()

            # avoid delivering loss from h_t and c_t
            # thus need to remove them from the computation graph
            encoder_ht, encoder_ct = encoder_ht.detach(), encoder_ct.detach()
            decoder_ht, decoder_ct = decoder_ht.detach(), decoder_ct.detach()

            # avoid gradient explosion
            _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), flags.gradients_norm)
            _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), flags.gradients_norm)

            # update parameters with optimizers
            encoder_optimizer.step()
            decoder_optimizer.step()

            losses.append(loss_value)

            if iteration % print_every == 0:
                percent = iteration / (flags.num_epochs * len(train_loader))
                time_since = time.time() - start_time
                time_remaining = time_since / percent - time_since
                print('Epoch: {}/{}'.format(e+1, num_epochs),
                      'Iteration: {}'.format(iteration),
                      'Time: {:.2f}m (- {:.2f}m)'.format(time_since/60, time_remaining/60),
                      'Loss: {}'.format(loss_value))
                logger.info('Epoch: {}/{} Iteration: {} Loss: {}'.format(e+1, num_epochs, iteration, loss_value))

            if iteration % save_every == 0:
                np.save("fig/losses.npy",losses)
                plt_loss(losses)
                try:
                    bleu = evaluation(dev_set,train_set.src_lang,train_set.dst_lang,encoder,decoder)
                except:
                    bleu = None
                logger.info("BLEU score: {}".format(bleu))
                torch.save(encoder,
                           '{}/encoder-{}.pth'.format(flags.checkpoint_path,iteration))
                torch.save(decoder,
                           '{}/decoder-{}.pth'.format(flags.checkpoint_path,iteration))

        # learning rate decay
        encoder_scheduler.step()
        decoder_scheduler.step()

    print("Time:{}s".format(time.time()-start_time))
    torch.save(encoder,'{}/encoder-final.pth'.format(flags.checkpoint_path))
    torch.save(decoder,'{}/decoder-final.pth'.format(flags.checkpoint_path))


# In[ ]:


from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
# https://cloud.tencent.com/developer/article/1042161

def evalOne(in_text, out_text, src_lang, dst_lang, encoder, decoder, beam_search=False, beam_width=2, print_flag=False):
    in_text, in_text_len = in_text
    if PAD_token in in_text[:in_text_len] or PAD_token in out_text or len(in_text) > flags.seq_size or len(out_text) > flags.seq_size:
        return None, None, -1
    encoder.eval() # set in evaluation mode
    decoder.eval()

    x = torch.tensor(in_text).to(device).reshape(1,-1)
    seq_len = torch.tensor([in_text_len]).to(torch.int64).to(device)
    # encoder
    encoder_ht, encoder_ct = encoder.initHidden(1)
    encoder_outputs, (encoder_ht, encoder_ct) = encoder(x, (encoder_ht, encoder_ct), seq_len)

    decoder_input = torch.tensor([BOS_token] * 1).reshape(1,1).to(device) # <BOS> token
    decoder_ht, decoder_ct = encoder_ht, encoder_ct # use last hidden state from encoder

    # decoder
    # run through decoder one time step at a time
    max_len = int(flags.seq_size*1.5)
    decoder_attentions = torch.zeros(max_len,flags.seq_size)
    if not beam_search:
        decoded_words = []
        decoded_index = []
        for t in range(max_len):
            if decoder.name == "Toy":
                decoder_output, (decoder_ht, decoder_ct) = decoder(decoder_input, (decoder_ht, decoder_ct))
            elif decoder.name == "Attn":
                decoder_output, (decoder_ht, decoder_ct), decoder_attn = decoder(decoder_input, (decoder_ht, decoder_ct), encoder_outputs)
                decoder_attentions[t] = decoder_attn.transpose(1,2).squeeze(0).squeeze(0).cpu().data
#                 print(sum(decoder_attentions[t]))
            else:
                decoder_output, (decoder_ht, decoder_ct), decoder_attention = decoder(decoder_input, (decoder_ht, decoder_ct), encoder_outputs)
                decoder_attentions[t,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data
            # choose top word from output
            top_value, top_index = decoder_output.data.topk(1)
            ni = top_index[0][0].item()
            decoded_index.append(ni)
            word = dst_lang.index2word[ni]
            decoded_words.append(word)
            if word == "<EOS>":
                break
            decoder_input = torch.LongTensor([ni]).reshape(1,1).to(device)
    else:
        """
        Beam seach:
        https://medium.com/@dhartidhami/beam-search-in-seq2seq-model-7606d55b21a5
        """
        path = [(BOS_token,0,[])] # input, value, words on the path
        for t in range(max_len):
            new_path = []
            flag_done = True
            for decoder_input, value, indices in path:
                if decoder_input == EOS_token:
                    new_path.append((decoder_input,value,indices))
                    continue
                elif len(path) != 1 and decoder_input in [BOS_token,PAD_token]:
                    continue
                flag_done = False
                decoder_input = torch.tensor([decoder_input]).reshape(1,1).to(device)
                if decoder.name == "Toy":
                    decoder_output, (decoder_ht, decoder_ct) = decoder(decoder_input, (decoder_ht, decoder_ct))
                elif decoder.name == "Attn":
                    decoder_output, (decoder_ht, decoder_ct), decoder_attn = decoder(decoder_input, (decoder_ht, decoder_ct), encoder_outputs)
                    decoder_attentions[t] = decoder_attn.transpose(1,2).cpu().data
                else:
                    decoder_output, (decoder_ht, decoder_ct), decoder_attention = decoder(decoder_input, (decoder_ht, decoder_ct), encoder_outputs)
    #                 decoder_attentions[t,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data
                # choose top word from output
                softmax_output = F.log_softmax(decoder_output,dim=2) # dim 2!
                top_value, top_index = softmax_output.data.topk(beam_width)
                top_value = top_value.cpu().squeeze().numpy() + value
                top_index = top_index.cpu().squeeze().numpy()
                for i in range(beam_width):
                    ni = int(top_index[i])
                    new_path.append((ni,top_value[i],indices+[ni]))
            if flag_done:
                _, value, decoded_index = new_path[0]
                break
            else:
                new_path.sort(key=lambda x:x[1]/len(x[2]),reverse=True) # normalization
                path = new_path[:beam_width]

        if not flag_done:
            _, value, decoded_index = path[0]
        decoded_words = []
        for ni in decoded_index:
            word = dst_lang.index2word[ni]
            decoded_words.append(word)

    pad_index = np.where(out_text == PAD_token)
    if len(pad_index[0]) == 0:
        pad_index = len(out_text)
    else:
        pad_index = pad_index[0][0]
    filter_outtext = list(filter("<PAD>".__ne__,out_text[:pad_index]))
    decoded_index = list(filter("<PAD>".__ne__,decoded_index))
    sm = SmoothingFunction()
    bleu = sentence_bleu([filter_outtext],decoded_index,smoothing_function=sm.method4)
    if print_flag:
        print(out_text[:pad_index])
        print(decoded_index)
        print("Bleu score: {}".format(bleu))
        res_words = " ".join(decoded_words)
        print("< {}".format(src_lang.getSentenceFromIndex(in_text)))
        print("= {}".format(dst_lang.getSentenceFromIndex(filter_outtext)))
        print("> {}".format(res_words))
        print()
    return decoded_words, decoder_attentions[:t+1, :flags.seq_size], bleu


# In[ ]:


def evaluation(dataset, src_lang, dst_lang, encoder, decoder, beam_search=False, beam_width=2, all_print_flag=False):
    start_time = time.time()
    bleus = []
    for i,(in_text, out_text) in enumerate(dataset):
        in_text = src_lang.getSentenceIndex(in_text,0,False)
        in_text_len = len(in_text)
        in_text = src_lang.padIndex(in_text,flags.seq_size)
        if len(in_text) == 0:
            continue
        out_text = dst_lang.getSentenceIndex(out_text,0,False)
        if all_print_flag:
            print_flag = True if i % 50 == 0 else False
        else:
            print_flag = False
        res_words, attention, bleu = evalOne((in_text,in_text_len),out_text,src_lang,dst_lang,encoder,decoder,beam_search=beam_search,beam_width=beam_width,print_flag=print_flag)
        if res_words != None:
            bleus.append(bleu)
    avg_bleu = np.mean(bleus)
    print("Evaluation time: {:.2f}s BLEU Score: {}".format(time.time()-start_time,avg_bleu))
    return avg_bleu


# In[ ]:


train_set = TextDataset("train",10000,max_seq_len=flags.seq_size,batch_size=flags.batch_size,trunc=flags.trunc)


# In[ ]:


if flags.trunc == -1:
    _, _, dev_set = preprocess("dev",10000)
    _, _, test_set = preprocess("test",10000)
    dataset = {"train_set":train_set,
               "dev_set":dev_set,
               "test_set":test_set}
else: # trunc
    dataset = {"train_set":train_set,
               "dev_set":train_set.pairs,
               "test_set":train_set.pairs}

encoder = EncoderRNN(train_set.src_lang.n_words, flags.lstm_size).to(device)
decoder = AttnDecoderRNN(flags.lstm_size,train_set.dst_lang.n_words).to(device)
# encoder = torch.load("checkpoint/encoder-final.pth")
# decoder = torch.load("checkpoint/decoder-final.pth")
train(encoder,decoder,dataset,batch_size=flags.batch_size,num_epochs=flags.num_epochs,print_every=flags.print_every,save_every=flags.save_every)


# In[ ]:


import matplotlib.ticker as ticker

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


# In[ ]:


evaluation(dataset["test_set"], dataset["train_set"].src_lang, dataset["train_set"].dst_lang, encoder, decoder, beam_search=True, beam_width=2, all_print_flag=True)


# In[ ]:


# src_lang = dataset["train_set"].src_lang
# dst_lang = dataset["train_set"].dst_lang
# for i,(in_text, out_text) in enumerate(dataset["test_set"]):
# #     if i == 5:
# #         break
#     in_text = src_lang.getSentenceIndex(in_text,0,False)
#     in_text_len = len(in_text)
#     in_text = src_lang.padIndex(in_text,flags.seq_size)
#     if len(in_text) == 0:
#         continue
#     out_text = dst_lang.getSentenceIndex(out_text,0,False)
#     res_words, attention, bleu = evalOne((in_text,in_text_len),out_text,src_lang,dst_lang,encoder,decoder,beam_search=False,beam_width=2,print_flag=False)


# In[ ]:


# src_lang = dataset["train_set"].src_lang
# dst_lang = dataset["train_set"].dst_lang
# for i,(in_text, out_text) in enumerate(dataset["train_set"].pairs):
#     if i < 105:
#         continue
#     if i == 110:
#         break
#     in_text = src_lang.getSentenceIndex(in_text,0,False)
#     in_text_len = len(in_text)
#     print(in_text,in_text_len)
#     in_text = src_lang.padIndex(in_text,flags.seq_size)
#     if len(in_text) == 0:
#         continue
#     out_text = dst_lang.getSentenceIndex(out_text,0,False)
#     res_words, attention, bleu = evalOne((in_text,in_text_len),out_text,src_lang,dst_lang,encoder,decoder,beam_search=False,beam_width=2,print_flag=True)
#     show_attention(" ".join([str(i) for i in range(in_text_len)]),res_words,attention[:len(res_words),:in_text_len])


# In[ ]:


# src_lang = dataset["train_set"].src_lang
# dst_lang = dataset["train_set"].dst_lang
# bleus = []
# for i,(in_text, out_text) in enumerate(dataset["train_set"].pairs):
#     if i % 100 == 0:
#         print(i)
#     in_text = src_lang.getSentenceIndex(in_text,0,False)
#     in_text_len = len(in_text)
#     in_text = src_lang.padIndex(in_text,flags.seq_size)
#     if len(in_text) == 0:
#         continue
#     out_text = dst_lang.getSentenceIndex(out_text,0,False)
#     res_words, attention, bleu = evalOne((in_text,in_text_len),out_text,src_lang,dst_lang,encoder,decoder,beam_search=False,beam_width=2,print_flag=False)
#     if res_words != None:
#         bleus.append(bleu)
# avg_bleu = np.mean(bleus)
# print("BLEU Score: {}".format(avg_bleu))

