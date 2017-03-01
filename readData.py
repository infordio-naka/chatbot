import os
import re
import numpy as np
from   keras.preprocessing.text import Tokenizer

datadir = os.path.join(os.getcwd(), "dataset/")

"""
text = ["Hello, World!\n", "HELLO, KERAS!\n", "Hello, Hello,\n"]
tk = Tokenizer(filters="\n\r", lower=True)
tk.fit_on_texts(text)
tk.fit_on_sequences(text)
print(tk.word_counts)
print(tk.word_docs)
print(tk.word_index)

print(tk.texts_to_sequences(text))
print(tk.texts_to_matrix(text, mode="tfidf"))
exit()
"""

def count_empty(sequence):
    cnt = 0
    for s in sequence:
        if (s==' '):
            cnt += 1
    return (cnt)

def splitEOS(data):
    edata  = []
    tmp   = ""
    for c in data: # get to "<EOS>"
        tmp += (c+" ")
        if (c=="<EOS>"):
            edata.append(tmp)
            tmp = ""
    return (edata)

def addPAD(data, max_length):
    pdata = []
    for s in data: # add "<PAD>"
        if (count_empty(s)<max_length):
            s+=" <PAD>"*(max_length-count_empty(s))
        pdata.append(s)

    return (pdata)

def getSequenceLength(data):
    # get sequence length (count empty)
    seq_length = []
    for s in data:
        seq_length.append(count_empty(s))
    max_length = max(seq_length)

    return (max_length)

def read_dataset(filename):
    fn   = os.path.join(datadir, filename)
    # open file
    fd   = open(fn, 'r')
    data = fd.readlines()
    fd.close()
    del data[-1]
    ptr2 = re.compile(r'[!|?|.]$')
    ptr  = re.compile(r'[!|?|.][\n|\r]')
    redata = []
    for s in data:
        redata.append(re.split(' ', re.sub(r'["-]', '', ptr.sub('{0} <EOS>'.format([ptr2.search(s).group() if ptr2.search(s) else ""][0]), s))))

    fdata = [flatten for inner in redata for flatten in inner]
    edata = splitEOS(fdata)
    max_length = getSequenceLength(edata)
    data  = addPAD(edata, max_length)

    # preprocessor(text)
    tk   = Tokenizer(filters="\r\n,-", lower=True)
    tk.fit_on_texts(data)

    return (tk, tk.texts_to_sequences(data))
    
def toNumpy(data):
    tmp  = np.zeros((len(data)+1, len(data[0])+1), dtype=np.int64)
    for d1 in range(len(data)):
        for d2 in range(len(data[d1])):
            tmp[d1][d2] = data[d1][d2]

    ret = np.reshape(tmp, (tmp.shape[0], 1, tmp.shape[1]))
    ret = np.delete(ret, -1, axis=0)
    ret = np.delete(ret, -1, axis=2)

    return (ret)

def splitTimesteps(data, step=100):
    d = []
    for i in range(0, len(data)-step+1):
        d.append(data[i:step*(i+1)])

    return(np.asarray(d))

def splitData(data, late=0.3):
    data_length  = len(data)
    split_length = int(data_length*late)
    #print(data_length)
    #print(split_length)

    s_data1 = data[split_length:]
    s_data2 = data[:split_length]
    
    return (np.asarray(s_data1), np.asarray(s_data2))


def splitXY(data):
    x = data[:-1]
    y = data[1:]
    
    return (x, y)

def toNHotvec(data, word_dict):
    word_val = list(word_dict.values())
    
    nhot_vec = np.zeros((data.shape[0], data.shape[2], len(word_val)))

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if (data[i][0][j] == 0):
                data[i][0][j] = 1
            nhot_vec[i][j][word_val.index(data[i][0][j])]=1

    return (nhot_vec)

if (__name__ == "__main__"):
    filename    = "subtitles.txt"
    tk, data    = read_dataset(filename)
    data        = toNumpy(data)
    train, test = splitData(data, late=0.3)
    train_x, train_y = splitXY(train)
    train_y = toNHotvec(train_y, tk.word_index)

