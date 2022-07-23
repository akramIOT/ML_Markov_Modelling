# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:31:12 2020
@author: Akram Sheriff
- Markov chain modelling for energy optimization by predicting optimum State Transition in  Wireless Mesh Networks
"""
import numpy as np
import pandas as pd
from random import seed
from random import random
import matplotlib.pyplot as plt
import seaborn as sns

# 6LLN Node  State Transition Probability Matrix  or Transformation Matrix
# The State Transition Probability Matrix is claimed Markovian or MC based as the state change of a Wireless 6LLN node is dependent
# on the present state rather than the past (History of past states).Its a memoryless stochastic random process in other sense.

P = np.array([[0.2, 0.7, 0.1],
              [0.9, 0.0, 0.1],
              [0.2, 0.8, 0.0]])

stateChangeHist= np.array([[0.0,  0.0,  0.0],
                          [0.0, 0.0,  0.0],
                          [0.0, 0.0,  0.0]])

# Initial State of a Wireless 6Lo/CGMesh Node
state=np.array([[1.0, 0.0, 0.0]])
currentState=0
stateHist=state
dfStateHist=pd.DataFrame(state)
distr_hist = [[0,0,0]]
seed(4)

# With Past state information and  Transition Matrix 'P', Dot Product computation is done here to predict the future state of a Node
def display_markov_plot():
     P = np.array([[0.2, 0.7, 0.1],
                   [0.9, 0.0, 0.1],
                   [0.2, 0.8, 0.0]])
     state = np.array([[1.0, 0.0, 0.0]])
     stateHist = state
     dfStateHist = pd.DataFrame(state)
     distr_hist = [[0,0,0]]
     for x in range(50):
          state=np.dot(state,P)
          print(state)
          stateHist=np.append(stateHist,state,axis=0)
          dfDistrHist = pd.DataFrame(stateHist)

     dfDistrHist.plot()

# This multinomail function is used to distribute the data and function is called in for Loop below
def simulate_multinomial(vmultinomial):
    # Simulate from multinomial distribution
    r = np.random.uniform(0.0, 1.0)
    CS = np.cumsum(vmultinomial)
    CS=np.insert(CS,0,0)
    m=(np.where(CS<r))[0]
    nextState=m[len(m)-1]
    return nextState

for x in range(1000):
    currentRow=np.ma.masked_values((P[currentState]), 0.0)
    nextState=simulate_multinomial(currentRow)
    # Keep track of node state changes
    stateChangeHist[currentState,nextState]+=1
    # Keep track of the state vector itself
    state=np.array([[0,0,0]])
    state[0,nextState]=1.0
    # Keep track of buffer state history
    stateHist=np.append(stateHist,state,axis=0)
    currentState=nextState
    # calculate the actual distribution over the 3 states (Rx, Tx, Sleep State)so far for 6LLN Nodes.
    totals=np.sum(stateHist,axis=0)
    gt=np.sum(totals)
    distrib =totals/gt
    # reshaping or  Flattening the data
    distrib = np.reshape(distrib,(1,3))
    distr_hist = np.append(distr_hist,distrib,axis=0)

print(distrib)
P_hat=stateChangeHist/stateChangeHist.sum(axis=1)[:,None]
# Check estimated state transition probabilities for different  6LLN  Nodes  based on history so far:
print(P_hat)
dfDistrHist = pd.DataFrame(distr_hist)
# Plot the distribution as the simulation progresses over time

final_state_matrix = sns.lineplot(data=dfDistrHist, markers= True)
final_state_matrix.set(xlabel='epoch/iterations', ylabel='MC State Transition Probability convergence', title='6LoWPAN/CG-Mesh Markov chain based Node Simulation History')
#dfDistrHist.plot(title="6LoWPAN/CG-Mesh Markov chain based Node Simulation History")
plt.show()


## Main Driver code
if __name__ == "__main__":
    display_markov_plot()

