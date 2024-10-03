import scipy.stats as stats
import numpy as np

# sim env based on random walk

class SimEnv:
    def __init__(self,timeout=1000):
        self.timeout = timeout
        self.steps_elapsed = 0
        self.s = np.random.uniform(low=-2, high=2)
        self.psi = np.array([0.6,0.6,0.4])
        self.neg_error = 0.1
        self.pos_error = 0.1
        self.pos_end = 3
        self.neg_end = -3
    def reset(self):
        self.steps_elapsed = 0
        self.s = np.random.uniform(low=-2, high=2)
        return self.s

    def termination(self):
        if self.s >= self.pos_end or self.s <= self.neg_end:
            done = True
        else:
            done = False
        return done

    def step(self,action):
        if action<0:
            error = self.neg_error
        else :
            error = self.pos_error 
        if action == -1:
            u = np.random.uniform(0,1,size = None)
            if u < self.psi[0]:
                ds = -2
            else:
                ds = 0
            mean = self.s + ds
        elif action == 0:
            u = np.random.uniform(0,1,size = None)
            if u < self.psi[1]:
                ds = 0
            else:
                ds = 2
            mean = self.s + ds
        else:
            u = np.random.uniform(0,1,size = None)
            if u < self.psi[2]:
                ds = 0
            else:
                ds = 2
            mean = self.s + ds
        sprime = np.random.normal(loc=mean, scale=error, size=None)
        reward = self.get_reward(sprime)
        self.s = sprime
        done = self.termination()
        self.steps_elapsed += 1
        return self.s, reward, done or self.steps_elapsed >= self.timeout, []

    def get_reward(self,sprime):
        if sprime >= self.pos_end:
            r = 0
        
        elif sprime <= self.neg_end:
            r = 0
        
        else:
            if sprime<0:
                r = -2
            else:
                r = -1.8
        return r