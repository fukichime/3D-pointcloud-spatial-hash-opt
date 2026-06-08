"""
Reference: Teschner et al. (2003) "Optimized Spatial Hashing for Collision Detection of Deformable Objects"

"""
import numpy as np
import torch

PRIMES = {
    'p1': 73856093,
    'p2': 19349663,
    'p3': 83492791
}

PRIMES2 = {
    'q1': 123456791,
    'q2': 987654321,
    'q3': 555555557
}

def spatial_hash(x, y, z, table_size, p1=73856093, p2=19349663, p3=83492791):

    h = (x * p1) ^ (y * p2) ^ (z * p3)
    return h % table_size

def expected_collision_probability(k, T):
    if T <= 0 or k <= 1:
        return 0.0
    
    k = float(k)
    T = float(T)
    
    exponent = -(k * (k - 1)) / (2 * T)
    result = 1 - np.exp(exponent)
    result = max(0.0, min(1.0, result))

    return result

def actual_collision_rate(indices):
    if len(indices) == 0:
        return 0.0
    unique = len(torch.unique(indices))
    return 1.0 - unique / len(indices)

def double_hash(x, y, z, table_size, max_probes=5):

    p1, p2, p3 = PRIMES['p1'], PRIMES['p2'], PRIMES['p3']
    q1, q2, q3 = PRIMES2['q1'], PRIMES2['q2'], PRIMES2['q3']
    
    h1 = (x * p1) ^ (y * p2) ^ (z * p3)
    h2 = (x * q1) ^ (y * q2) ^ (z * q3)
    
    indices = []
    used = set()
    
    for i in range(len(x) if hasattr(x, '__len__') else 1):
        if hasattr(x, '__len__'):
            slot = (h1[i] % table_size)
            step = 0
            while slot in used and step < max_probes:
                slot = (slot + h2[i]) % table_size
                step += 1
            used.add(slot)
            indices.append(slot)
        else:
            slot = (h1 % table_size)
            step = 0
            while slot in used and step < max_probes:
                slot = (slot + h2) % table_size
                step += 1
            used.add(slot)
            indices.append(slot)
    
    return indices if len(indices) > 1 else indices[0]