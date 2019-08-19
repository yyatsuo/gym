#!/usr/bin/env python3
import gym
import numpy as np
import random
import time

GAMMA = 0.9
ALPHA = 0.6
EPSILON_0 = 0.5
NUM_EPISODES = 5000
NUM_STEPS = 200
NUM_DIGITIZED = 6

max_state = 0

def bins(clip):
    return np.linspace(-1*clip, clip, NUM_DIGITIZED)

def digitized_state(observation):
    global max_state
    pos, v, angle, angle_v = observation
    digitized = [
        np.digitize(pos, bins=bins(2.4)),
        np.digitize(v, bins=bins(3.0)),
        np.digitize(angle, bins=bins(0.21)),
        np.digitize(angle_v, bins=bins(3.0))
    ]
    num = sum([x * (NUM_DIGITIZED**i) for i, x in enumerate(digitized)])
    if num > max_state:
        max_state = num
    return num

def update_qtable(table, state, action, reward, next_state):
    next_q_max = max(table[next_state][0], table[next_state][1])
    table[state, action] = (1-ALPHA) * table[state, action] + ALPHA * (reward + GAMMA * next_q_max)
    return table

def next_action(next_state):
    if epsilon < random.uniform(0,1):
        return np.argmax(q_table[next_state])
    else:
        return np.random.choice([0,1])


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    print(env.action_space.n)
    q_table = np.random.uniform(low=-1, high=1, size=(1555, env.action_space.n))
    reward_rambda = 0
    is_learned = False
    is_done = False
    for episode in range(NUM_EPISODES):
        total_reward = 0
        epsilon = EPSILON_0 * (1/(episode+1))
        state = digitized_state(env.reset())
        action = np.argmax(q_table[state])

        if is_done:
            break

        for t in range(NUM_STEPS):
            if is_learned:
                epsilon = 0
                env.render()
                time.sleep(0.1)

            observation, reward, done, info = env.step(action)
            if done:
                if t < NUM_STEPS-1:
                    reward = -200
                else:
                    reward = 1
            else:
                reward = 1
            total_reward += reward

            next_state = digitized_state(observation)
            q_table = update_qtable(q_table, state, action, reward, next_state)
            action = next_action(next_state)
            state = next_state

            if done:
                if is_learned:
                    is_done = True
                reward_rambda = total_reward*0.25 + reward_rambda*0.75
                print("Episode {0} finished after {1} timesteps with reward {2}".format(episode,t+1, total_reward))
                if(reward_rambda > 199.9):
                    print("Learned!")
                    is_learned = True
                break

    print("END")
    env.close()
