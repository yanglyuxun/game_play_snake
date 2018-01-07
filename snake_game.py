#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snake game

@author: Lyuxun Yang
"""
import numpy as np
from numpy.random import choice
import random

def rand_choice(size, snake):
    ''' choose a random point when knowing some points
    have been taken'''
    avail = [(x,y) for x in range(size[0]) 
        for y in range(size[1])
        if (x,y) not in snake]
    return random.choice(avail) if avail else None # if no choice, return None

def move_map(xy,d):
    ''' from (x,y) move to direction d'''
    if d==0: return xy[0]-1,xy[1] #UP
    elif d==1: return xy[0]+1,xy[1] #DOWN
    elif d==2: return xy[0],xy[1]-1 #LEFT
    elif d==3: return xy[0],xy[1]+1 #RIGHT
    else: raise 'wrong direction'

oppo_map = {0:1, 1:0, 2:3, 3:2}

class snake_API(object):
    ''' the snake game logic'''
    def __init__(self, size=(5,5)):
        self.size = size
    def first(self):
        '''first step'''
        # init snake head
        self.snake = [(choice(self.size[0]),choice(self.size[1]))] 
        # init fruit position
        self.fruit = rand_choice(self.size, self.snake)
        # init direction
        self.d = choice(4) # direction of moving
        # init score
        self.score = 0
        return self.snake, self.fruit, oppo_map[self.d]
    def move(self, d):
        '''move a step'''
        if (d is not None) and d != oppo_map[self.d]:
            self.d = d # otherwise, d unchanged
        x,y = move_map(self.snake[0], self.d)
        if x<0 or y<0 or x>=self.size[0] or y>=self.size[1]: #out of bound
            return None, None, None, 'd'  #DEAD
        elif (x,y) in self.snake[:-1]: # touch self
            return None, None, None, 'd' # DEAD
        elif (x,y) == self.fruit: # eat fruit
            self.score += 1
            self.snake.insert(0, self.fruit)
            # init new fruit
            self.fruit = rand_choice(self.size, self.snake)
            if self.fruit:
                result = 'e' #EAT
            else: # win the whole game!
                result = 'w' #Win
        else: #normal move
            self.snake.insert(0,(x,y))
            self.snake.pop(-1)
            result = 'n' # normal move
        return self.snake, self.fruit, oppo_map[self.d], result
    def show(self):
        ''' show the current state '''
        board = np.zeros(self.size, dtype = np.str)
        x,y = self.snake[0]
        board[x,y] = 'X'
        for x,y in self.snake[1:]:
            board[x,y] = 'o'
        x,y = self.fruit
        board[x,y] = 'F'
        print(np.where(board=='',' ',board))

class snack_pygame(object):
    '''pygame snake controler'''
    def __init__(self,rows,cols):
        import pygame
        pygame.init()
        self.s = pygame.display.set_mode((20*cols, 20*(rows+1)))
        pygame.display.set_caption('Snake')
        self.fruit = pygame.Surface((20, 20))
        self.fruit.fill((0, 0, 255))
        self.head = pygame.Surface((20, 20))
        self.head.fill((255,0,0))
        self.tail = pygame.Surface((20, 20))
        self.tail.fill((255, 110, 0))
        self.font = pygame.font.SysFont('Arial', 20)
        #self.clock = pygame.time.Clock()
        self.s.fill((255, 255, 255))
        self.update = pygame.display.update
        self.quit = pygame.quit
        self.update()
    def show(self, snake, fruit):
        if snake is None: return
        score = len(snake) -1
        self.s.fill((255, 255, 255))
        x,y = snake[0]; x*=20; x+=20; y*=20
        self.s.blit(self.head, (y,x))
        for x,y in snake[1:]:
            x *= 20; x+=20; y*=20
            self.s.blit(self.tail, (y,x))
        x,y = fruit; x *= 20; x+=20; y*=20
        self.s.blit(self.fruit, (y,x)) 
        t=self.font.render(str(score), True, (0, 0, 0))
        self.s.blit(t, (0, 0))
        self.update()
    def close(self):
        self.quit()