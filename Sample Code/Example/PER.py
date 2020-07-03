import random
import numpy as np
from SumTree import SumTree

class Memory:
    eps   = 0.01
    alpha = 0.6
    beta  = 0.4
    beta_increment_per_sampling = 0.001
    absolute_error_upper        = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        
    def add(self, experience):
        
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        if max_priority == 0:
            max_priority = self.absolute_error_upper
            
        self.tree.add(max_priority, experience)

    def sample(self, n):
        
        data_batch = []
        idxs       = []
        priorities = []
        segment    = self.tree.total() / n
        self.beta  = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            
            a     = segment * i
            b     = segment * (i + 1)
            value = random.uniform(a, b)
            
            (idx, p, data) = self.tree.get(value)
            priorities.append(p)
            data_batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(n * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return data_batch, idxs, is_weight
        
    def update(self, idx, errors):
        errors += self.eps
        clipped_errors = np.minimum(errors, self.absolute_error_upper)
        priorities = np.power(clipped_errors, self.alpha)

        for ti, p in zip(idx, priorities):
            self.tree.update(ti, p)
        
        
        
        
        
        
        
        
        
        
        
        
        