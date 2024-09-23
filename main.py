import os
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0", render_mode="human")
# env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2_000

SHOW_EVERY = 100

# since high and low normally have same length, just choose either one
# define the number of discrete bins for each dimension of the observation space
# dos here means 20 bins for position and 20 bins for velocity
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_SIZE = np.array(DISCRETE_OS_SIZE)
# dows calculate the size of each bin
# it represents the size (range) of each bin for each dimension
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# randomness
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE.tolist() + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    state = state[0] if isinstance(state, tuple) else state
    # does this discreate state means how many bin the current state's position in respect to the lowest position
    # can be divided?
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    discrete_state = np.clip(discrete_state, 0, DISCRETE_OS_SIZE - 1)
    return tuple(discrete_state.astype(int))


for episode in range(EPISODES):
    print(episode)
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    state = env.reset()
    discrete_state = get_discrete_state(state)

    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # Correctly unpacking the four returned values
        new_state, reward, done, info = env.step(action)

        # If you need 'truncated', check if it's inside 'info'
        truncated = info.get('TimeLimit.truncated', False)  # Example of handling 'truncated' if needed

        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        # print(f"Reward: {reward}, New_state: {new_state}")
        done = done or truncated

        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"The cart made it to the goal on episode {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    average_reward = sum(ep_rewards[:]) / len(ep_rewards[:])
    aggr_ep_rewards['ep'].append(episode)
    aggr_ep_rewards['avg'].append(average_reward)
    aggr_ep_rewards['min'].append(min(ep_rewards[:]))
    aggr_ep_rewards['max'].append(max(ep_rewards[:]))

    output_directory = "qtables"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    np.save(f"{output_directory}/{episode}-qtable.npy", q_table)

    print(f"Episode: {episode}, Average: {average_reward}, Min: {min(ep_rewards[:])}, Max: {max(ep_rewards[:])}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()
