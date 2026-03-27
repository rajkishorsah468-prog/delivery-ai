import numpy as np

class DeliveryEnv:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal = [self.size-1, self.size-1]
        return self.state()

    def state(self):
        return tuple(self.agent_pos)

    def step(self, action):
        if action == 0:
            self.agent_pos[1] -= 1
        elif action == 1:
            self.agent_pos[1] += 1
        elif action == 2:
            self.agent_pos[0] -= 1
        elif action == 3:
            self.agent_pos[0] += 1

        reward = -1
        done = False

        if self.agent_pos[0] < 0 or self.agent_pos[1] < 0 or \
           self.agent_pos[0] >= self.size or self.agent_pos[1] >= self.size:
            reward = -5
            self.reset()

        if self.agent_pos == self.goal:
            reward = 10
            done = True

        return self.state(), reward, done, {}