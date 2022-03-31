# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:07:15 2022

@author: Julie
"""
#On va utiliser la méthode d'apprentissage Q-learning
import gym
import numpy as np 

# 1. Charge l'environement
env = gym.make('FrozenLake8x8-v1')
Q = np.zeros([env.observation_space.n,env.action_space.n])# Q est une fonction de valeur état-action


# 2. Parametres de l'apprentissage Q-learning
eta = .628
gma = .9
epis = 5000
rev_list = [] # liste de récompenses

# 3. Algorithme de Q-learning 
for i in range(epis):
    # Reset environement
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #l'algorithme Q-Table
    while j < 99:
        env.render()
        j+=1
        #On choisit une action dans le tableau Q 
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Récupère un nouvel état et récompense de l'environement
        s1,r,d,_ = env.step(a)
        #On met à jour Q avec l'apprentissage
        Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    rev_list.append(rAll)
    env.render()
    
print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
print("Final Values Q-Table")
print(Q)

