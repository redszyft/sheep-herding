import numpy as np
from sklearn.metrics import pairwise

def make_herd(n_sheep, origin_x, origin_y, max_radius):
    angles = np.random.rand(n_sheep) * 2 * np.pi
    radii = np.random.rand(n_sheep) * max_radius
    x = origin_x + radii * np.cos(angles)
    y = origin_y + radii * np.sin(angles)
    sheep_pos = np.vstack([x,y]).T

    sheep_orient = np.vstack([np.cos(angles), np.sin(angles)]).T
    return sheep_pos, sheep_orient

def herd_alignment(orientations, nns_idx):
    result = np.zeros(orientations.shape)
    for i, group in enumerate(nns_idx):
        result[i] = orientations[group].sum(axis=0) / len(group)
    return result

def herd_attraction(positions, nns_idx):
    result = np.zeros(positions.shape)
    for i, group in enumerate(nns_idx):
        result[i] = positions[group].sum(axis=0) / len(group)
    return result

def dUa(positions, la):
    """Herd inner repulsion"""
    
    relative_pos = positions[:, np.newaxis, :] - positions[:,:]
    distances = pairwise.euclidean_distances(positions, positions)
    distances += np.eye(len(positions)) * 1e3 # prevents 0 division and self-interaction
    result = relative_pos * np.exp(-distances[:,:,np.newaxis] / la) / distances[:,:,np.newaxis]
    return result.sum(axis=0)

def dUs(shepherd_r, positions, ls):
    """Shepherd-herd repulsion"""
    
    relative_pos = shepherd_r - positions
    distances = np.linalg.norm(relative_pos, axis=1)
    result = relative_pos * np.exp(-distances[:,np.newaxis] / ls) / distances[:,np.newaxis]
    return result