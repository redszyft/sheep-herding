import numpy as np

def candidate_moves(dog_v, n_moves):
    dog_fi = np.random.rand(n_moves) * 2 * np.pi
    return  dog_v * np.vstack([np.cos(dog_fi), np.sin(dog_fi)]).T