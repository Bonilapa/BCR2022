import gym
import numpy as np

class Network:
    def __init__(self, env):
        self.pvariance = 0.1
        self.nhiddens = 5
        self.ninputs = env.observation_space.shape[0]
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            self.noutputs = env.action_space.shape[0]
        else:
            self.noutputs = env.action_space.n


    def initparameters(self):
        self.w1 = np.random.randn(self.nhiddens, self.ninputs) * self.pvariance
        self.w2 = np.random.randn(self.noutputs, self.nhiddens) * self.pvariance
        self.b1 = np.zeros(shape=(self.nhiddens, 1))
        self.b2 = np.zeros(shape=(self.noutputs, 1))

    def update(self, observation):
        observation.resize(self.ninputs, 1)
        z1 = np.dot(self.w1, observation) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = np.tanh(z2)
        if(isinstance(env.action_space, gym.spaces.box.Box)):
            action = a2
        else:
            action = np.argmax(a2)
        return action

    def evaluate(self, env, nepisodes, show = False):
        fitness = 0
        observation = env.reset()
        for i_episode in range(nepisodes):
            print("\n\n--------------Episode start------------")
            for t in range(100):
                if (show):
                    env.render()
                action = self.update(observation)
                observation, reward, done, info = env.step(action)

                fitness += reward
                if done:
                    print("Finished after {} timesteps".format(t+1))
                    break


        return fitness / nepisodes




env = gym.make("CartPole-v0")
nepisodes = 20
network = Network(env)
for t in range(10):
    network.initparameters()
    fitness = network.evaluate(env, nepisodes)
    print("Result fitness: ", str(fitness))
env.close()

