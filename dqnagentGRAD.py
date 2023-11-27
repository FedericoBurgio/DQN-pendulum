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


class DQN:
            
    def __init__(self,env):
        
        #Environment
        self.env=env 
        self.gamma=.99
        self.epsilon=1
        self.numberEpisodes=125
        self.maxIter=100
        self.stateDimension=2 #POSSO PRENDERE LE DIMENSIONI DALL ENV??
        self.actionDimension=1
        self.NoOfActions=11
        self.possibleActions = np.linspace(-2, 2, self.NoOfActions)
        self.possibleActionsIndex=np.arange(0,self.NoOfActions,1)
    
        
        # Buffer
        self.replayBufferSize=100000
        self.replayBuffer=deque(maxlen=self.replayBufferSize)
        self.batchReplayBufferSize=64
        
        #Target update
        self.UpdatePeriod=20
        self.UpdateCounter=0
        
        #Q network model creation
        self.Q=self.create_model6()
        self.Q_target=self.create_model6()
        
       # self.Q.load_weights('grad.h5')
        self.Q_target.set_weights(self.Q.get_weights())
        
        #Plot reasons
        self.q_values_history = []
        self.reward_history=[]
        self.chosen_actions_history = []
        
    def create_model6(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.stateDimension, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(11, activation='linear'))
        model.compile(optimizer='adam', loss='mse')#default alpha is 1e-3 ->ok
        return model 
    
    def trainingEpisodes(self):
        
        # episodes loop
        self.Q.summary()
        
#        Model: "sequential"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #   
#=================================================================
# dense (Dense)               (None, 64)                192       
#                                                                 
# dense_1 (Dense)             (None, 32)                2080      
#                                                                 
# dense_2 (Dense)             (None, 11)                363       
#                                                                 
#=================================================================
#Total params: 2,635
#Trainable params: 2,635
#Non-trainable params: 0
#_________________________________________________________________
        
        print("trainingEpisode called")
        for i in range(self.numberEpisodes):
            
            s=self.env.reset()
            
            rewardsEpisode=[]
            self.chosen_actions_history=[]
            
            for j in range(self.maxIter): 
                #s : current state
                #a : action
                #r : reward (cost)
                #s_ : next state (given a)
                
                
                a = self.act(s)
                 
                s_, r = self.env.step(self.possibleActions[a])          
                self.chosen_actions_history.append(self.possibleActions[a])
                rewardsEpisode.append(r)
                
                # add experience to the replay buffer
                self.replayBuffer.append((s,a,r,s_))
                
                #network training if replay buffer has enough elements
                if (len(self.replayBuffer)>self.batchReplayBufferSize):                    
                    self.trainNetwork()
                
                if j % 20 == 0:  # record Q-values every 20 steps
                    q_values = self.Q.predict(np.array([s]))[0]
                    self.q_values_history.append(q_values)
                    
                # update the current state!
                s=s_
             
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Episode no. {}".format(i))
            print("epsilon: "+str(self.epsilon))
            self.epsilon *= 0.96 
            self.epsilon = max(0.01, self.epsilon)
            #self.epsilon=.9
            print("Sum of current episode rewards {}".format(np.sum(rewardsEpisode)))
            print("Mean of all episodes rewards {}".format(np.mean(self.reward_history)))
            
            
            self.reward_history.append(np.sum(rewardsEpisode))
            self.plotQValues()
            self.plotRewards()
            self.plotChosenActions()

    
    def act(self, s): #epsilon greedy action (not actual action, rather its index)
            if uniform() < self.epsilon: 
                return np.random.choice(self.possibleActionsIndex)
            else:
                return np.argmax(self.Q.predict(np.array([s])))#max->min
                #MAXMIN
               
  
    def trainNetwork(self):
        # sample a batch from the replay buffer
        miniBatch = random.sample(self.replayBuffer, self.batchReplayBufferSize)
        
        # here we form current state batch and next state batch
        currentStateBatch = np.zeros(shape=(self.batchReplayBufferSize, 2))
        nextStateBatch = np.zeros(shape=(self.batchReplayBufferSize, 2))
            
        for i, (s, _, _, s_) in enumerate(miniBatch):
            currentStateBatch[i, :] = s.reshape(-1)
            nextStateBatch[i, :] = s_.reshape(-1)

        # convert to TensorFlow tensors
        currentStateBatch = tf.convert_to_tensor(currentStateBatch, dtype=tf.float32)
        nextStateBatch = tf.convert_to_tensor(nextStateBatch, dtype=tf.float32)

        # use GradientTape for automatic differentiation
        with tf.GradientTape(persistent=True) as tape:
            # predict Q-values for the current and next state using the main network
            QcurrentStateMainNetwork = self.Q(currentStateBatch)
            QnextStateTargetNetwork = self.Q_target(nextStateBatch)

            # initialize the loss
            loss = 0

            for i, (_, a, r, _) in enumerate(miniBatch):
                y = r + self.gamma * tf.reduce_max(QnextStateTargetNetwork[i])

                # gather the Q-values for the selected action
                Q_current_state = QcurrentStateMainNetwork[i]
                Q_current_state_action = tf.gather(Q_current_state, a)

                # compute the mean squared error loss
                loss += tf.square(y - Q_current_state_action)

        # compute gradients
        gradients = tape.gradient(loss, self.Q.trainable_variables)

        # perform the optimization step
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer.apply_gradients(zip(gradients, self.Q.trainable_variables))

        # clean up the tape
        del tape

        # after UpdatePeriod training sessions, update the coefficients 
        # of the target network
        self.UpdateCounter += 1  
        if self.UpdateCounter >= self.UpdatePeriod:
            # copy the weights to targetNetwork
            self.Q_target.set_weights(self.Q.get_weights())        
            # reset the counter
            self.UpdateCounter = 0



    def evaluate(self, num_episodes=5):
        total_rewards = []
        
        for episode in range(num_episodes):
            s = self.env.reset()
            episode_reward = 0
            self.chosen_actions_history=[]
            for _ in range(100):
                action = np.argmax(self.Q.predict(np.array([s])))
                self.chosen_actions_history.append(self.possibleActions[action])
                s_, r = self.env.step(self.possibleActions[action])
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

    def saveW(self,name='grad.h5'):
        print()
        self.Q.save_weights(name)
        return

from pendulum import Pendulum

agent=DQN(Pendulum())

agent.evaluate()

training=True
if training:
    agent.trainingEpisodes()

