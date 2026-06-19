import numpy as np
from sklearn.neighbors import NearestNeighbors
from dog import candidate_moves
from herd import make_herd, herd_alignment, herd_attraction, dUa, dUs

def cost_func(target, positions, dog_r, ls, w_mean, w_std, w_col):
    cm = positions.sum(axis=0) / len(positions)
    dist2target = np.linalg.norm(target - cm)
    cohesion = np.power(np.linalg.norm(positions, axis=1) - np.linalg.norm(cm), 4).sum() / len(positions)
    cohesion = np.power(cohesion, 0.25)
    collinear = np.linalg.norm(dog_r - cm + ls * (target - cm))
    return w_mean * dist2target + w_std * cohesion + w_col * collinear
    
n_steps = 2000

target = np.array([10,2])

n_sheep = 30
v_a = 0.04 # sheep speed
gamma = 0.005 # sheep2sheep attraction
la = 0.01 # typical sheep size
r_attr = np.sqrt(n_sheep) * la # sheep attraction radius
beta = 0.1 # sheep repulsion strength
r_align = 0.1 # sheep alignment radius

n_attemps = 20
ls = 0.3 # dog influence scale
delta = 0.9 # dog repulsion strength
dog_v = 20 * v_a

w_std = 5
w_mean = 1
w_col = 1e-3

nn = NearestNeighbors(leaf_size=1)

def run_sim():
    dog_r = np.array([0., 0.])
    dog_tracker = np.zeros((n_steps, 2))
    dog_tracker[0] = dog_r.copy()

    sheep_pos, sheep_orient = make_herd(n_sheep, 3, 2, max_radius=0.5)
    sheep_pos_tracker = np.zeros((n_steps, sheep_pos.shape[0], sheep_pos.shape[1]))
    sheep_pos_tracker[0] = sheep_pos.copy()
    sheep_orient_tracker = np.zeros((n_steps, sheep_orient.shape[0], sheep_orient.shape[1]))
    sheep_orient_tracker[0] = sheep_orient.copy()

    for i in range(1, n_steps):
        moves = candidate_moves(dog_v, n_attemps)
        costs = np.zeros(n_attemps)
        tmp_sheep_pos = np.zeros((n_attemps, n_sheep, 2))
        nn.fit(sheep_pos)
        
        for j, move in enumerate(moves):
            tmp_sheep_pos[j] = sheep_pos.copy()
            tmp_dog_r = dog_r + move
    
            _, attr_idx = nn.radius_neighbors(sheep_pos, r_attr)
            _, align_idx = nn.radius_neighbors(sheep_pos, r_align)
            tmp_sheep_pos[j] += v_a * (herd_alignment(sheep_orient, align_idx) \
                                    - gamma * herd_attraction(sheep_pos, attr_idx)) \
            - beta * dUa(sheep_pos, la) \
            - delta * dUs(tmp_dog_r, sheep_pos, ls)
    
            costs[j] = cost_func(target, tmp_sheep_pos[j], tmp_dog_r, ls, w_mean, w_std, w_col)
    
        best_move = np.argmin(costs)
        dog_r += moves[best_move]
        dog_tracker[i] = dog_r.copy()
        
        sheep_pos = tmp_sheep_pos[best_move]
        sheep_orient = sheep_pos - sheep_pos_tracker[i-1]
        sheep_orient /= np.linalg.norm(sheep_orient, axis=1)[:, np.newaxis]
        
        sheep_pos_tracker[i] = sheep_pos.copy()
        sheep_orient_tracker[i] = sheep_orient.copy()
        
        if all(np.linalg.norm(target-sheep_pos, axis=1) < 0.3):
            print('Success at step: ', i)
            break

    return dog_tracker , sheep_pos_tracker
        