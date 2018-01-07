
# coding: utf-8
'''
The new train method

@author: Lyuxun Yang
'''

import keras 
from keras import backend as K
from keras.models import Model
from keras.initializers import TruncatedNormal
from keras.layers import Input, Embedding, Dense, Dropout, Flatten, Conv2D, MaxPooling2D


SHOW = False # Show the game


rows = 5
cols = 5 # the size of game board
save_dir  = './%i_%i/'%(rows,cols)
ACTIONS = 4 # actions: 4 directions
OBSERVATION = 1000 # how many observations before start to train
INITIAL_EPSILON = 0.5 # the prob of random trial. it will decrease until
FINAL_EPSILON = 0.01 # ...until this number
EXPLORE = 5000 # how many episodes from INITIAL_EPSILON to FINAL_EPSILON
REPLAY_MEMORY = 3000 # number of states to remember for replaying
BATCH = 100 # size of a minibatch
GAMMA = 0.99 # the decay rate

def buildmodel(show_model=False):
    input1 = Input(shape=(rows*cols,2),dtype='float32')
#    x = Conv2D(16,(2,2),strides=(1,1),padding='same',
#               activation='relu',
#               kernel_initializer=TruncatedNormal(),
#               bias_initializer='zeros')(input1)
    flat1 = Flatten()(input1)
#    input2 = Input(shape=[2],dtype='float32',name='in2')
#    x = keras.layers.concatenate([flat1,input2])
    x = Dense(64,kernel_initializer=TruncatedNormal(),
              bias_initializer='zeros',activation='relu')(flat1)
    #x = Dropout(0.5)(x) large data set, no need for dropout
    x = Dense(32,kernel_initializer=TruncatedNormal(),
              bias_initializer='zeros',activation='relu')(x)
    x = Dense(ACTIONS,kernel_initializer=TruncatedNormal(),
                   bias_initializer='zeros')(x)
    model = Model(input1, x)
    model.compile(loss='mse',optimizer='adam')
    if show_model: print(model.summary())
    return model




