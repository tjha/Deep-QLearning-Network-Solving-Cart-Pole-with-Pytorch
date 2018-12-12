import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size, C=1):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 1.0   
        self.epsilon_max = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0001
        self.epsilon = 0.5 
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target = self._build_target()
        self.C = C

        self.c_counter = 0

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_target(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state, episode):
        self.epsilon = self.get_epsilon(episode)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) 
    def get_epsilon(self, t):
        return self.epsilon_max - min(self.epsilon_decay * t, self.epsilon_max - 0.01)
    def replay(self, batch_size):
        minibatch = []
        searching = min(len(self.memory), batch_size)
        for _ in range(searching):
            p = random.randint(0,len(self.memory) - 1)
            minibatch.append(self.memory[p])
        # minibatch = np.random.choice(
        #     self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.target.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            self.c_counter += 1
            if self.c_counter % self.C == 0:
                self.copyWeights()
                self.c_counter = 0
    def copyWeights(self):
        self.target.set_weights(self.model.get_weights())

    

def CartPole(env, state_size, action_size, c):
    
    agent = DQNAgent(state_size, action_size, C=c)

    episodes = 5000

    scores = deque(maxlen=100)

    mean_scores = []
    batch_episodes = []


    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        sum_reward = 0
        for time_t in range(500):

            action = agent.act(state, e+1)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            sum_reward += reward
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, sum_reward))
                scores.append(sum_reward)
                break
        
        if (e+1)% 100 == 0:
            mean_scores.append(np.mean(scores))
            batch_episodes.append(e)
            print(np.mean(scores))
            if np.mean(scores) >= 195:
                break



        # train the agent with the experience of the episode
        agent.replay(32)

    plt.plot(batch_episodes, mean_scores)
    plt.ylabel("average score over past 100 episodes")
    plt.xlabel("episode")
    plt.savefig('testCar.png')

def MountainCar(env):
    c = 100
    CartPole(env, 2, 3, c)
        
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    CartPole(env, 4, 2, 100)
    env = gym.make('MountainCar-v0')
    MountainCar(env)
