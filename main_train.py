#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
the main parts of the training

@author: Lyuxun Yang
"""

##########
# This is the file to get all hyperparameters and model
from train5_5 import *
##########


import numpy as np
from collections import deque
from numpy import random
import pickle
import os,time, sys
from snake_game import snake_API, snack_pygame

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

####### if you run at the first time:
if not os.path.exists(save_dir+'model.h5'):
    model = buildmodel(True)
    mem = deque(maxlen = REPLAY_MEMORY) # store the memories
    records = deque(maxlen = REPLAY_MEMORY) # only record numbers
    epsilon = INITIAL_EPSILON
    t=0
    scores = []
else:####### if you continue to run:
    with open(save_dir+'mem.pickle','rb') as f:
        (t,epsilon,mem,scores)=pickle.load(f)
    from keras.models import load_model
    model = load_model(save_dir+'model.h5')
    #epsilon = INITIAL_EPSILON # use this line when you want to enlarge the old epsilon
from keras import backend as K
print('lr0:',K.get_value(model.optimizer.lr))
K.set_value(model.optimizer.lr, 0.0001) # used to adjust lr
print('lr1:',K.get_value(model.optimizer.lr))
##################################

## log file
if not os.path.exists(save_dir+'log.csv'):
    with open(save_dir+'log.csv','a') as f:
        f.write('t,score,loss\n')
def logging(t,score,loss):
    with open(save_dir+'log.csv','a') as f:
        f.write('%i,%i,%f\n'%(t,score,loss))   

def make_input(snake, fruit):
    '''get data from the api and make the data for NN model'''
    inp = np.zeros((rows*cols, 2)) 
    assert len(snake)+1 <= rows*cols
    xh,yh = snake[0] # head
    inp[0] = [xh,yh]
    xf,yf = fruit # fruit
    inp[1] = [xf-xh, yf-yh]
    for i,(x,y) in enumerate(snake[1:]):
        inp[2+i] = [x-xh, y-yh]
    return np.expand_dims(inp,0)


g = snake_API(size=(rows,cols))
snake, fruit, wrongd = g.first()
if SHOW: pyg = snack_pygame(rows,cols)
if SHOW: pyg.show(snake, fruit)
loss = -1 # a init value
oppo_map = {0:1, 1:0, 2:3, 3:2}

while True: # start to loop
    t += 1
    #### make an action
    inp = make_input(snake,fruit) # the input for NN
    valid = [i for i in range(4) if i!=wrongd]
    if random.random()<=epsilon:
        a_t = random.choice(valid)
    else:
        qs = model.predict(inp)[0,valid]
        a_t = valid[np.argmax(qs)]
        
    #### forward one step
    snake1, fruit1, wrongd1, result = g.move(a_t)
    if SHOW: pyg.show(snake1, fruit1)
    if result == 'n': # noraml
        mem.append((inp,a_t,make_input(snake1,fruit1),-50))
    elif result == 'e': # eat
        mem.append((inp,a_t,make_input(snake1,fruit1),1000))    
    elif result == 'd': # die
        mem.append((inp,a_t,None,-1000))
        scores.append(g.score) # save the score
        snake1, fruit1, wrongd1 = g.first() # start a new game
        if SHOW: pyg.show(snake1, fruit1)
    elif result == 'w': # win
        mem.append((inp,a_t,None,2000))
        scores.append(g.score) # save the score
        snake1, fruit1, wrongd1 = g.first() # start a new game
        if SHOW: pyg.show(snake1, fruit1)
    else:
        raise 'wrong'
    snake, fruit, wrongd = snake1, fruit1, wrongd1 #update
    
    #### update epsilong
    if epsilon > FINAL_EPSILON and t > OBSERVATION:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    if epsilon < FINAL_EPSILON: epsilon = FINAL_EPSILON
    
    #### trian the model if observations are enough
    if t > OBSERVATION:
        # sample a minibatch
        minibatch = random.choice(len(mem), min(BATCH,len(mem)))
        # initialize input and target
        inputs = np.zeros((BATCH, rows*cols,2))
        targets = np.zeros((BATCH, ACTIONS))
        # fill them
        for i,j in enumerate(minibatch):
            nninp, nna, nninp1, nnr = mem[j]
            inputs[i:i+1] = nninp
            targets[i] = model.predict(nninp)
            if nninp1 is not None: # if not die or win
                Qt1 = model.predict(nninp1)[0,[i for i in range(4) if i!=oppo_map[nna]]]
                targets[i,nna] = nnr + GAMMA * np.max(Qt1)
            else:
                targets[i,nna] = nnr
        # train the model
        loss = model.train_on_batch(inputs,targets)
        #print(loss)
        
    #### save the model 
    if t%10==0: 
        print('t=%i, try=%i, epsilon=%f, loss=%f'%(t,len(scores),epsilon,loss))
        if scores: print('score=%i, max=%i'%(scores[-1],np.max(scores)))
    if t%1000==0:
        print('saving model...',end=' ')
        model.save(save_dir+'model.h5')
        with open(save_dir+'mem.pickle','wb') as f:
            pickle.dump((t,epsilon,mem,scores),f)
        print('Done.')
