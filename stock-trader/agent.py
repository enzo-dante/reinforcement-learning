from collections import deque
import random
import numpy as np

import tensorflow as tf

#### build rl model
class RL_TRADER:
    ### define class initializer that is called each time we define an obj of class
    # connects atr to a specific obj instance of class
    # trader actions = stay, buy, sell
    def __init__(self, state_size, action_space=3, model_name="RL_TRADER"):
        ### define hperparams of nn
        self.state_size = state_size
        self.action_space = action_space
        # define exp replay (how many els stored for batch sampling)
        self.memory = deque(maxlen=2000)
        # stock holder
        self.inventory = []
        self.model_name = model_name

        # gamma param which helps nn focus on rewards that are more important to getting reward(state farther from reward = less important)
        self.gamma = 0.95

        # epsilon: how many times model chooses random action (env exploration)
        self.epsilon = 1.0
        # over time want model to take trained action, but periodically still explore
        self.epsilon_final = 0.01
        # epsilon rate that goes from random to trained actions
        self.epsilon_decay = 0.995
        # create nn and initialize
        self.model = self.modelBuilder()

    ### define neural network for trading bot
    def modelBuilder(self):
        model = tf.keras.models.Sequential()
        # states = end days and stock prices of day

        # units= # of neurons in layer
        model.add(
            tf.keras.layers.Dense(
                units=32, activation="relu", input_dim=self.state_size
            )
        )
        model.add(tf.keras.layers.Dense(units=64, activation="relu"))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        ### add model output layer
        # of units should equal # of classes
        # linear = mean sqr error
        model.add(tf.keras.layers.Dense(units=self.action_space, activation="linear"))

        ### compile model
        # mse = mean sqr error
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001))
        return model

    # take state as input, return action based on state
    # determine if random action or use model to perform action
    def trade(self, state):
        if random.random() <= self.epsilon:
            # randrange = limits random # to argument range (0, 1, 2)
            return random.randrange(self.action_space)
        # model will choose action to perform based on input state; if random > epsilon
        actions = self.model.predict(state)
        # return action with highest probability
        return np.argmax(actions[0])

    # custom training function
    # take batch of saved data (xp replay) & train maodel based on batch
    def batchTrain(self, batch_size):
        # select data from xp replay memory
        batch = []
        # iterate through memory and randomly select from most recent exp
        # end index = len(self.memory)
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        # iterate through batch arr
        # order of vars important
        for state, action, reward, next_state, done in batch:
            reward = reward
            # validate agent not in terminal state
            # if agent in terminal state, cur_reward = reward
            if not done:
                # use bellman equation to get reward
                # max = maximum value of all actions in a given state
                # V(s) = max(R(s,a) + gamma * V(s'))
                reward = reward + self.gamma * np.amax(
                    # 0 = output size
                    self.model.predict(next_state)[0]
                )

            target_action = self.model.predict(state)
            target_action[0][action] = reward

            #### TRAIN THE MODEL
            # fit(my_input, target/real_answers, # of epochs model is trained on entire dataset)
            # epoch set to 1 because we want to train model from new sample from our batch
            # verbose = 0 (don't print trianing results)
            self.model.fit(state, target_action, epochs=1, verbose=0)

        # decrease epsilon (random actions taken epsilon % of the time)
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

