#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record the playing procedure to gif

@author: Lyuxun Yang
"""

import pickle
import moviepy.editor as mpy
from keras.models import load_model
from snake_game import snake_API, snack_pygame
from numpy import random
import numpy as np
import pygame,os

rows, cols = 5,5
frame = 200
fps = 3
save_dir = './%i_%i/'%(rows, cols)

with open(save_dir+'mem.pickle','rb') as f:
    (t,epsilon,mem,scores)=pickle.load(f)
model = load_model(save_dir+'model.h5')

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

if not os.path.exists('./tmp'):
    os.mkdir('./tmp')


g = snake_API(size=(rows,cols))
snake, fruit, wrongd = g.first()
pyg = snack_pygame(rows,cols)
pyg.show(snake, fruit)
pygame.image.save(pyg.s, './tmp/'+'0.jpeg')
loss = -1 # a init value
oppo_map = {0:1, 1:0, 2:3, 3:2}
for i in range(frame):
    inp = make_input(snake,fruit) # the input for NN
    valid = [i for i in range(4) if i!=wrongd]
    if random.random()<=epsilon:
        a_t = random.choice(valid)
    else:
        qs = model.predict(inp)[0,valid]
        a_t = valid[np.argmax(qs)]
    snake1, fruit1, wrongd1, result = g.move(a_t)
    if result in ['d','w']: #die or win
        snake1, fruit1, wrongd1 = g.first() # start a new game
    pyg.show(snake1, fruit1)
    pygame.image.save(pyg.s, './tmp/'+str(i+1)+'.jpeg')
    snake, fruit, wrongd = snake1, fruit1, wrongd1
pyg.close()


#### convert images to gif
fname = save_dir+'play.gif'
flist = os.listdir('./tmp')
flist.sort(key=lambda x: int(x.split('.')[0]))
flist = ['./tmp/'+f for f in flist]
movie = mpy.ImageSequenceClip(flist, fps=fps)
movie.write_gif(fname, fps=fps)

#### delete images
for f in flist:
    os.remove(f)
os.removedirs('./tmp/')
