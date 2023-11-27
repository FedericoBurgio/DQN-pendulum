#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:55:06 2023

@author: fede
"""

import numpy as np
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from collections import deque 
import tensorflow as tf
from numpy.random import uniform
import matplotlib.pyplot as plt
from keras.models import load_model

import time

from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model   


Motor_steps=11

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.regularizers import l2


class DQN:
            
    def __init__(self,env):
    
        #Agent        
        self.gamma=.99
        self.epsilon=1
        self.numberEpisodes=160
        self.maxIter=200

        #Environment
        self.env=env 
        self.stateDimension=4 #POSSO PRENDERE LE DIMENSIONI DALL ENV??
        #self.actionDimension=1
        self.NoOfActions=Motor_steps^2 #(7^2)
        
        self.possibleActions = np.linspace(-self.env.umax, self.env.umax, 11)
        #self.possibleActionsIndex=np.arange(0,self.NoOfActions,1)
    
        
        # Buffer
        self.replayBufferSize=100_000
        self.replayBuffer=deque(maxlen=self.replayBufferSize)
        self.batchReplayBufferSize=64
        
        #Target update
        self.UpdatePeriod=120#era 80, pare ci sia una dipendenza tra update period e influenza short term delle ultime azioni 
        self.UpdateCounter=0
        
        #Q networks model creation
        self.Q=self.create_model49()
        self.Q_target=self.create_model49()
        
        #Model weights loading
        #self.Q.load_weights('test2.h5')
        #self.Q.load_weights('testFit.h5')
        self.Q_target.set_weights(self.Q.get_weights())
        
        #Plot reasons
        self.q_values_history = []
        self.reward_history=[]
        self.chosen_actions_history = []
    
        # Time tracking
        self.start_time = time.time()

        
    def create_model49(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.stateDimension, activation='relu'))#128->256
        model.add(Dense(64, activation='relu'))#64->128
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(11, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model


    def trainingEpisodes(self):
        
        # episodes loop
        self.Q.summary()
        
        print("trainingEpisode called")
        for i in range(self.numberEpisodes):
            start_episode_time = time.time()
            rewardsEpisode=[]
            s=self.env.reset(fixed=False)
            self.chosen_actions_history=[]
            
            for j in range(self.maxIter): 
                #s : current state
                #a : action
                #r : reward (cost)
                #s_ : next state (given a)
                                      
                
                a = self.act(s)
               
                s_, r = self.env.step([self.possibleActions[a],0]) 
     
                self.chosen_actions_history.append(self.possibleActions[a])
                rewardsEpisode.append(r)
                
                #add experience to the replay buffer
                self.replayBuffer.append((s,a,r,s_))
                 
                #network training if replay buffer has enough elements
                if (len(self.replayBuffer)>self.batchReplayBufferSize):                    
                    self.trainNetwork()
#                
#                if j % 20 == 0:  # record Q-values every 20 steps
#                    q_values = self.Q.predict(np.array([s]))[0]
#                    self.q_values_history.append(q_values)
#                    
                # update s
                s=s_
             
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Episode no. {}".format(i))
            print("epsilon: "+str(self.epsilon))
            self.epsilon *= 0.96
            self.epsilon = max(0.005, self.epsilon)
           # self.epsilon=.9
            print("Sum of current episode rewards {}".format(np.sum(rewardsEpisode)))
            print("Mean of all episodes rewards {}".format(np.mean(self.reward_history)))
            episode_elapsed_time = time.time() - start_episode_time
            print("Episode Elapsed Time: {:.2f} seconds".format(episode_elapsed_time))
            
            self.reward_history.append(np.sum(rewardsEpisode))
            #self.plotQValues() # non funziona con doppio pendolo 
            self.plotRewards()
            self.plotChosenActions()

    
    
    
    def act(self, s): #epsilon greedy action (not actual action, rather its index)
            if uniform() < self.epsilon: 
                #return np.random.choice(self.possibleActionsIndex)
                return random.randint(0,10)
            else:
                return np.argmax(self.Q.predict(np.array([s])))
            
            
    def trainNetwork(self):

        # sample a batch from the replay buffer
        miniBatch=random.sample(self.replayBuffer, self.batchReplayBufferSize)
        
        # here we form current state batch 
        # and next state batch
        # they are used as inputs for prediction
        currentStateBatch=np.zeros(shape=(self.batchReplayBufferSize,4))
        nextStateBatch=np.zeros(shape=(self.batchReplayBufferSize,4))            
        # this will enumerate the tuple entries of the miniBatch
        # index will loop through the number of tuples  
           
        for i,(s,_,_,s_) in enumerate(miniBatch):
            # first entry of the tuple is the current state
            currentStateBatch[i,:]=s.reshape(-1)
            # fourth entry of the tuple is the next state
            nextStateBatch[i,:]=s_.reshape(-1)
             
        

        # here, use the target network to predict Q-values 
        QnextStateTargetNetwork=self.Q_target.predict(nextStateBatch)
        # here, use the main network to predict Q-values 
        QcurrentStateMainNetwork=self.Q.predict(currentStateBatch)
        
        # now, we form batches for training
        # input for training
        inputNetwork=currentStateBatch
        # output for training
        outputNetwork=np.zeros(shape=(self.batchReplayBufferSize,11))
        outputNetwork=np.array(outputNetwork)
        
        for i,(_,a,r,_) in enumerate(miniBatch):
            
            y=r+self.gamma*np.max(QnextStateTargetNetwork[i])
            
  
            # this actually does not matter since we do not use all the entries in the cost function
#            outputNetwork[i]=QcurrentStateMainNetwork[i]
            outputNetwork[i] = QcurrentStateMainNetwork[i]
            # this is what matters
            outputNetwork[i,a]=y
        
        # here, we train the network
        
        #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        #self.Q.fit(inputNetwork, outputNetwork, batch_size=self.batchReplayBufferSize, verbose=0, epochs=2, validation_split=0.2, callbacks=[early_stopping])
        self.Q.fit(inputNetwork, outputNetwork, batch_size=self.batchReplayBufferSize, verbose=0, epochs=2)
      
        #self.grad(inputNetwork, outputNetwork)
        # after UpdatePeriod training sessions, update the coefficients 
        # of the target network
        # increase the counter for training the target network
        self.UpdateCounter+=1  
        if self.UpdateCounter>=self.UpdatePeriod:
            # copy the weights to targetNetwork
            self.Q_target.set_weights(self.Q.get_weights())        
            #print("Target network updated!")
            #print("Counter value {}".format(self.UpdateCounter))
            # reset the counter
            self.UpdateCounter=0
          


    def evaluate(self, num_episodes=5):
        total_rewards = []
        
        for episode in range(num_episodes):
            s = self.env.reset()
            episode_reward = 0
            self.chosen_actions_history=[]
            for _ in range(self.maxIter):
                action = np.argmax(self.Q.predict(np.array([s])))
                self.chosen_actions_history.append(self.possibleActions[action])
                s_, r = self.env.step([self.possibleActions[action],0])
                episode_reward += r
                s = s_
              
            total_rewards.append(episode_reward)
            loss_value = self.Q.evaluate(np.array([s]), np.array([[0]*11]),verbose=0)  # Assuming a dummy target here
            print(f"Episode {episode + 1}, Reward: {episode_reward}, Loss: {loss_value}")
            print("")   
            self.plotChosenActions()
        average_reward = np.mean(total_rewards)
        print(f"\nAverage Reward over {num_episodes} episodes: {average_reward}")
        
        return average_reward
    
    def plotQValues(self):
        q_values_history = np.array(self.q_values_history)
        for action_j in range(11):
            plt.plot(q_values_history[:, action_j], label=f' {round(self.possibleActions[action_j],1)}')
        plt.xlabel('Training Steps (Every 20 steps)')
        plt.ylabel('Q-values')
        plt.legend()
        plt.title('Q-values during Training')
        plt.show()
        
    def plotRewards(self):
        reward_h = np.array(self.reward_history)
        plt.plot(reward_h)
        plt.xlabel('episodes')
        plt.ylabel('cost')
        plt.title('cost during Training')
        plt.show()

    def plotChosenActions(self):
        chosen_actions = np.array(self.chosen_actions_history)
        plt.plot(chosen_actions)
        plt.xlabel('Training Steps')
        plt.ylabel('Chosen Action')
        plt.title('Chosen Action during Training')
        plt.show() 
    
    def saveW(self,name='test2.h5'):
        self.Q.save_weights(name)
        return

    def elapsed_time(self):
        """Return the elapsed time since the start."""
        return time.time() - self.start_time

from pendulum import Pendulum

def saveW(name='DoubleFit.h5'):
    agent.Q.save_weights(name)

agent=DQN(Pendulum(2))

agent.evaluate(0)


training=False
if training:
    agent.trainingEpisodes()
    print("Elapsed Time After Training:", agent.elapsed_time())
