import random
import numpy as np 
import scipy.stats as stats

# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py

class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class TruncatedNormalProcess(RandomProcess):
    def __init__(self, sigma=0.5, lower=0, upper=1, decay_rate=0.99,size=1):
        # self.mu = mu
        self.sigma_ori = sigma
        self.sigma = sigma
        self.lower = lower
        self.upper = upper
        self.decay_rate = decay_rate
        self.size = size

    def sample(self, mu):
        tn = stats.truncnorm((self.lower-mu) / self.sigma, (self.upper-mu) / self.sigma, loc=mu, scale=self.sigma) 
        samples = tn.rvs(self.size)
        return samples

    def step(self, reset=True):
        if reset:
            self.sigma = self.sigma_ori
        else:
            self.sigma = self.sigma * self.decay_rate

    def reset_states(self):
        self.sigma = self.sigma_ori

if __name__ == '__main__':
    r = TruncatedNormalProcess()
    for i in range(100):
        x = random.random()
        a = r.sample(x)
        print(x, a)
        r.step()
