from keras.models           import Sequential, load_model
from keras.layers.core      import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers       import RMSprop
from readData               import *
from model                  import CBModel
import os
import numpy as np
import time
import gc

filename = "subtitles.txt"
savedir  = os.path.join(os.getcwd(), "models/")

# config
nb_epoch       = 1
batch_size     = 32
vocab_size     = 1000
model_continue = False

# read Dataset
print("...read Dataset")
tk, data = read_dataset(filename)

word_dict = tk.word_index

data = toNumpy(data)

# prepare data
train, test = splitData(data, late=0.3)
#test, _ = splitData(test, late=0.9)
#print(len(train))
#print(len(test))

train_x, train_y = splitXY(train)
#train_x = toNHotvec(train_x, word_dict)
train_y = toNHotvec(train_y, word_dict, vocab_size)
#train_x          = splitTimesteps(train_x, step=1)
#print(sum(train_x[-1]))
#print(sum(train_y[-2]))
#print(len(train_y))
test_x, test_y   = splitXY(test)
#test_x           = splitTimesteps(test_x, step=1)
test_y  = toNHotvec(test_y, word_dict, vocab_size)
print("...Done!")

# load model or create model
input_shape  = train_x.shape
output_shape = train_y.shape
print("input_shape",  input_shape)
print("output_shape", output_shape)

cb           = CBModel()

if (model_continue):
    print("...load model")
    cb.load(savedir)
    print("...Done!")
else:
    print("...create new model")
    cb.create_model(input_shape, output_shape)
    print("...Done")

print("-------------------")
print("-   Train Start   -")
print("-------------------")
cb.train(train_x, train_y, batch_size, nb_epoch, verbose=1)

# save model
cb.save(sdir=savedir, overwrite=True)

# evaluate
cb.eval(test_x, test_y, verbose=0)

# predict
cb.predict(tk)

gc.collect()
