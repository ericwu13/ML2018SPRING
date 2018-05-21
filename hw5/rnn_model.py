import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

# build model
def RNN(args):
    inputs = Input(shape=(args.max_length, args.w2vdim))
    # RNN
    dropout_rate = args.dropout_rate
    RNN_cell = LSTM(int(args.hidden_size*1),
                    return_sequences=True,
                    dropout=0.5,
                    recurrent_dropout=0.5,
                    activation='tanh'
                    )(inputs)
    '''
    RNN_cell = GRU(int(args.hidden_size*1),
                    return_sequences=True,
                    dropout=0.5,
                    recurrent_dropout=0.5,
                    activation='tanh'
                    )(RNN_cell)
    '''
    RNN_cell = GRU(int(args.hidden_size*1),
                    return_sequences=False,
                    dropout=0.5,
                    recurrent_dropout=0.5,
                    activation='tanh'
                    )(RNN_cell)

    #RNN_output = RNN_cell(embedding_inputs)
    RNN_output = RNN_cell

    size = args.hidden_size*2
    # DNN layer
    outputs = Dense(size, kernel_regularizer=regularizers.l2(0), activation='selu')(RNN_output)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(dropout_rate)(outputs)

    size = size // 2
    outputs = Dense(size, kernel_regularizer=regularizers.l2(0), activation='selu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(dropout_rate)(outputs)

    size = size // 4
    outputs = Dense(size, kernel_regularizer=regularizers.l2(0), activation='selu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(dropout_rate)(outputs)

    size = size // 4
    outputs = Dense(size, kernel_regularizer=regularizers.l2(0), activation='selu')(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)

    model =  Model(inputs=inputs,outputs=outputs)

    # optimizer
    adam = Adam()
    print ('compile model...')
    model.compile(adam, 'binary_crossentropy', metrics=['accuracy'])
    # compile model

    return model
