import os
import gc
import numpy                      as     np
from   readData                   import splitEOS, addPAD, getSequenceLength
from   seq2seq.models             import SimpleSeq2Seq
from   keras.models               import Sequential, load_model
from   keras.layers.core          import Dense, Activation
from   keras.layers.recurrent     import LSTM
from   keras.layers.normalization import BatchNormalization
from   keras.layers.embeddings    import Embedding
from   keras.optimizers           import RMSprop

class CBModel(object):
    def __init__(self):
        self.model = None

    def create_model(self, input_shape, output_shape):
        self.model = Sequential()
        #self.model.add(Embedding(input_dim=input_shape[0], output_dim=input_shape[1]*2, input_length=input_shape[1]))
        #self.model.add(BatchNormalization())
        self.model.add(SimpleSeq2Seq(output_dim=output_shape[2], 
                                     output_length=output_shape[1], 
                                     input_shape=(input_shape[1], input_shape[2]), unroll=True))
        """
        self.model.add(LSTM(512, return_sequences=True, unroll=True))
        self.model.add(LSTM(1024, return_sequences=False, unroll=True))
        
        self.model.add(Dense(output_classes))
        self.model.add(Activation("softmax"))
        """
        self.model.compile(optimizer=RMSprop(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

    def train(self, train_x, train_y, batch_size, nb_epoch):
        self.model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch)

    def eval(self, test_x, test_y, verbose=0):
        score = self.model.evaluate(test_x, test_y, verbose=verbose)
        print(score)

    def predict(self, tk):
        i=0
        keys   = list(tk.word_index.keys())
        values = list(tk.word_index.values())
        while (True):
            pred_x = input("[you] >> ")
            if (pred_x==''):
                break
            pred_x += " <EOS>"
            max_length = getSequenceLength(pred_x)
            pred_x = addPAD([pred_x], 19)
            pred_x  = np.array([tk.texts_to_sequences(pred_x)])
            pred   = self.model.predict(pred_x, verbose=0)
            print(pred)
            """
            print("[bot] >> {0}".format(keys[values.index(pred[0])]))
            i += 1
            """

    def save(self, sdir, overwrite=True):
        self.model.save(os.path.join(sdir, "chatbot_model.h5"))
        self.model.save_weights(os.path.join(sdir, "chatbot_weights.h5"), overwrite=overwrite)

    def load(self, sdir):
        self.model = load_model(os.path.join(sdir, "chatbot_model.h5"))
        self.model.load_weights(os.path.join(sdir, "chatbot_weights.h5"))

if __name__ == "__main__":
    input_length = 1
    input_dim = 19
    
    output_length = 19
    output_dim = 6437

    samples = 6733

    x = np.random.random((samples, input_length, input_dim))
    y = np.random.random((samples, output_length, output_dim))
    cb = CBModel()
    input_shape  = x.shape
    output_shape = y.shape
    cb.create_model(input_shape, output_shape)
    cb.train(x, y, 1, 1)
    gc.collect()
