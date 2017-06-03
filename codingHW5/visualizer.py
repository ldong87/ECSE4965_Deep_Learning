#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:21:05 2017

@author: ldong
"""
import pickle as pk
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import json
import numpy as np

with open('word_embed.pkl','rb') as f:
  w_embed = pk.load(f)
  
with open('vocab.json','r') as f:
  vocab = json.load(f)
  
s = ["monday", "tuesday", "wednesday", "thursday", "friday",
"saturday", "sunday", "orange", "apple", "banana", "mango",
"pineapple", "cherry", "fruit"]
word_vec = [(i, vocab[i]) for i in s]

model = TSNE(n_components=2, random_state=0)
Y = model.fit_transform(w_embed)
wv = Y[np.array([item[1][0] for item in word_vec])]

plt.scatter(wv[:, 0], wv[:, 1], s=2)
for label, x, y in zip(s, wv[:, 0], wv[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',fontsize=5)
plt.savefig('word_viz.png',dpi=600)