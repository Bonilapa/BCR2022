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

'''
pendulum.py

1. What is encoded in observation vector?

2D Euclidean space coordinates and angular velocity, and their limits

2. What is encoded in action vector?

Pendulum torque and its limits

3. how the initial conditions are varied in env.reset?

starting angle is [-pi, pi] and starting angular velocity in [-1, 1]

4. How the reward is calculated?

-(theta^2 + 0.1*theta_dt^2 + 0.001*torque^2)

5. When the episode terminated?

After 200 steps, no other criteria

'''
