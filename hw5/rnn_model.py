import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

# build model

def FC(args):
    inputs = Input(shape=(30000,))

    outputs = Dense(512, activation='selu')(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(0.5)(outputs)

    outputs = Dense(512, activation='selu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(0.5)(outputs)

    outputs = Dense(128, activation='selu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(0.5)(outputs)

    outputs = Dense(1, activation='sigmoid')(outputs)
    model =  Model(inputs=inputs,outputs=outputs)

    adam = Adam()
    print ('compile model...')
    model.compile(adam, 'binary_crossentropy', metrics=['accuracy'])
    return model
def RNN(args):
    inputs = Input(shape=(args.max_length, args.w2vdim))
    # RNN
    dropout_rate = args.dropout_rate
    RNN_cell = LSTM(512, return_sequences=True,
                    dropout=0.5,
                    recurrent_dropout=0.5,
                    activation='tanh')(inputs)
    RNN_cell = GRU(512, return_sequences=False,
                    dropout=0.5,
                    recurrent_dropout=0.5,
                    activation='tanh')(RNN_cell)

    RNN_output = RNN_cell

    # FC layer
    outputs = Dense(1024, activation='selu')(RNN_output)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(0.5)(outputs)

    outputs = Dense(512, activation='selu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(0.5)(outputs)

    outputs = Dense(128, activation='selu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(0.5)(outputs)

    outputs = Dense(1, activation='sigmoid')(outputs)

    model =  Model(inputs=inputs,outputs=outputs)

    # optimizer
    adam = Adam()
    print ('compile model...')
    model.compile(adam, 'binary_crossentropy', metrics=['accuracy'])
    # compile model

    return model
