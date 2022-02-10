import gym

env = gym.make('CartPole-v0')

for i_episode in range(10):

    observation = env.reset()
    fitness = 0

    for t in range(200):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        # print("\n___"+str(action)+"___\n")
        observation, reward, done, info = env.step(action)
        fitness += reward
        # print("\nFitness: "+str(fitness)+"\n")

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("\nFitness: "+str(fitness)+"\n")

            break
env.close()
