import gymnasium as gym
import mdp
import numpy as np

# env = gym.make('Taxi-v3') --- The program freezes with this environment for some reason, and then not responding window pop up.
# Had to work with FrozenLake.

discount_factors = [0.5, 0.7, 0.9]
num_episodes = [50, 100]

for discount_factor in discount_factors:
    print(f"Discount factor: {discount_factor}")

    for num_episode in num_episodes:
        total_steps_list = []
        total_reward_list = []
        env = gym.make('FrozenLake-v1', render_mode='human')

        for _ in range(num_episode):
            done = False
            total_steps = 0
            total_reward = 0

            state, _ = env.reset()

            '''policy, _ = mdp.value_iteration(env,
                                            env.action_space.n,
                                            env.observation_space.n,
                                            discount_factor=discount_factor)'''
            policy, _ = mdp.policy_iteration(env,
                                             env.action_space.n,
                                             env.observation_space.n,
                                             discount_factor=discount_factor)
            while not done:
                action = np.argmax(policy[state])
                state, reward, done, _, _ = env.step(action)
                # env.render()
                total_steps += 1
                total_reward += reward

            total_steps_list.append(total_steps)
            total_reward_list.append(total_reward)

        avg_steps = np.mean(total_steps_list)
        avg_reward = np.mean(total_reward_list)
        print(f"Avg steps for {num_episode} episodes: {avg_steps}")
        print(f"Avg reward for {num_episode} episodes: {avg_reward}")
