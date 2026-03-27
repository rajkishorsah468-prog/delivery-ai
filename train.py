import numpy as np
import matplotlib.pyplot as plt
from env import DeliveryEnv

env = DeliveryEnv()
q_table = {}

episodes = 200
rewards = []

for _ in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        if state not in q_table:
            q_table[state] = np.zeros(4)

        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)

        if next_state not in q_table:
            q_table[next_state] = np.zeros(4)

        q_table[state][action] += 0.1 * (
            reward + 0.9 * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)

plt.plot(rewards)
plt.title("Learning Progress")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.savefig("output.png")
print("Graph saved as output.png")

import time
while True:
    time.sleep(100)