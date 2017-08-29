#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 21:20:42 2017

@author: zihao
"""

import numpy as np 

'''
input embedding
linear transform
tanh
output embedding
softmax
cross entropy
'''

def lookup(w,i):
    return w[i]

def linear(w,b,x):
    return np.dot(w,x)+b

def tanh(x):
    return np.tanh(x)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def cross_entropy(y,i):
    return -np.log(y[i])

    
# parameters: input_embedding_matrix, previous_word_index, linear_w, linear_b, output_emdedding_w , output_embedding_b, current_word_index
def forward(input_embedding_matrix, previous_word_index, linear_w, linear_b, output_emdedding_w , output_embedding_b, current_word_index):
    print("Forward")
    
    # input embedding
    h1 = lookup(input_embedding_matrix,previous_word_index)
    print("h1")
    print(h1)
    
    # linear transform
    h2 = linear(linear_w, linear_b, h1)
    print("h2")
    print(h2)
    
    # tanh
    h3 = tanh(h2)
    print("h3")
    print(h3)
    
    # output embedding
    h4 = linear(output_emdedding_w,output_embedding_b,h3)
    print("h4")
    print(h4)
    
    # softmax
    h5 = softmax(h4)
    print("h5")
    print(h5)
    
    # cross entropy
    ce = cross_entropy(h5,current_word_index)
    print("ce")
    print(ce)

def derivative_softmax(loss,i):
    djdx = loss
    djdx[i] -= 1
    return djdx
        
def derivative_linear(derivate_y,w,b,x):
    djdw = np.dot(derivate_y,x.T)
    djdx = np.dot(derivate_y,w.T) 
    djdb = derivate_y
    return djdw,djdb,djdx

def derivative_tanh(derivative_y,y):
    djdx = derivative_y*(1-y*y)
    return djdx

def derivative_input_embedding(derivative_y,m,i):
    djdm = np.zeros_like(m)
    djdm[i] = derivative_y
    return djdm

def backward():
    print("Backward")
    print()
    
    # backward
    # derivative softmax
    djdh4 = derivative_softmax(loss,i)
    print("djdh4")
    print(djdh4)
    
    # derivative output embedding
    djd_output_embed_w,djd_output_embed_b, djdh3= derivative_linear(djdh4,output_emdedding_w,output_embedding_b,h3)
    print("djd_output_embed_w, djd_output_embed_b,djdh3")
    print(djd_output_embed_w)
    print(djd_output_embed_b)
    print(djdh3)
    
    
    
    
    

def update_weight():
    pass

def main():
     # Define the matrix
    input_embed = np.array([[0.4,1],[0.2,0.4],[-0.3,2]])
    linear_w = np.array([[1.2,0.2],[-0.4,0.4]])
    linear_b = np.array([0,0.5])
    output_embed = np.array([[-1,1],[0.4,0.5],[-0.3,0.2]])
    output_embed_b = np.array([0,0.5,0])
    pre_word = 0 # a
    current_word = 1 # b
    eta = 0.1

if __name__ == "__main__":
    main()



    
    
    