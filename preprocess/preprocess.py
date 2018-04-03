'''

'''
import argparse
import pandas as pd
import sys
from functools import reduce
import pickle
from collections import defaultdict



def padding(sentence,max_length_sent):
    '''
    input is the index form of a document
    '''

    padded_sentence = sentence[:max_length_sent] if len(sentence) >= max_length_sent else sentence + [0] * (max_length_sent - len(sentence))
    return padded_sentence

def stat_label(labels):
    label_freq = defaultdict(int)
    for l in labels:
        label_freq[l] += 1
    print (label_freq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sent_length', type=int, default = 120,help="""the length of sentences truncated""")
    parser.add_argument('-train_prop', type=float, default = 0.95)
    args = parser.parse_args()


    data_frame = pd.read_csv('data/train_first.csv')
    predict_frame = pd.read_csv('data/predict_first.csv')

    ########## Step 1 ###########
    #过滤掉非汉字字符
    all_sents = []
    all_labels = []
    for index, row in data_frame.iterrows():
        row_values = row.values
        raw_text = row_values[1]
        label = row_values[2]
        review_text = ''.join(list(filter(lambda c: '\u4e00' <= c <= '\u9fa5',raw_text))) #过滤掉非汉字
        sent = review_text if len(review_text) < args.sent_length else review_text[:args.sent_length]
        all_sents.append(sent)
        all_labels.append(label)

    predict_sents = []
    predict_ids = []
    for index, row in predict_frame.iterrows():
        row_values = row.values
        ID = row_values[0]
        raw_text = row_values[1]
        #label = row_values[2]
        review_text = ''.join(list(filter(lambda c: '\u4e00' <= c <= '\u9fa5',raw_text))) #过滤掉非汉字
        sent = review_text if len(review_text) < args.sent_length else review_text[:args.sent_length]
        predict_sents.append(sent)
        predict_ids.append(ID)


    ########## Step 2 ###########
    #制作char2index和label2index字典，并存储为pickle。
    chars = set([])
    labels = set(all_labels)
    for sent in all_sents:
        chars = chars | set(sent)

    indices = range(2,len(chars)+2)
    char2index = {char: cid for char, cid in zip(chars,indices)}
    char2index['unk'] = 1
    char2index['pad'] = 0

    indices = range(len(labels))
    label2index = {label: lid for label, lid in zip(labels,indices)}

    pickle.dump(char2index,open('char2index.pickle','wb'))
    pickle.dump(label2index,open('label2index.pickle','wb'))


    ########## Step 3 ###########
    #将doc和label变成index, 然后pad,存储
    sents = []
    labels = []
    for sent,label in zip(all_sents,all_labels):
        label = label2index[label]
        idx = [char2index[c] if c in char2index else 1 for c in ' '.join(sent).split()]
        sents.append(padding(idx,args.sent_length))
        labels.append(label)

    length = len(labels)
    stat_label(labels[:int(length*args.train_prop)])
    pickle.dump((sents[:int(length*args.train_prop)],labels[:int(length*args.train_prop)]),open('train_preprocessed.pickle','wb'))
    pickle.dump((sents[int(length*args.train_prop):],labels[int(length*args.train_prop):]),open('test_preprocessed.pickle','wb'))
    pickle.dump((sents[:200],labels[:200]),open('tiny_preprocessed.pickle','wb'))

    ###### Predict #######
    sents = []
    ids  = []
    for sent,ID in zip(predict_sents,predict_ids):
        idx = [char2index[c] if c in char2index else 1 for c in ' '.join(sent).split()]
        sents.append(padding(idx,args.sent_length))
        ids.append(ID)
    pickle.dump((sents,ids),open('predict_preprocessed.pickle','wb'))
    ############## 存储参数，用于训练 ##########
    config = {}
    config['vocab_size'] = len(char2index)
    config['num_labels'] = len(label2index)
    config['sent_length'] = args.sent_length
    pickle.dump(config,open('config_preprocess.pickle','wb'))
