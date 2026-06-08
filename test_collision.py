import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.hash_function import expected_collision_probability

k = 703
test_Ts = [262144, 16384, 1024, 256]

print("Testing collision probability formula:")
print(f"k = {k}")
print("-" * 50)
for T in test_Ts:
    prob = expected_collision_probability(k, T)
    print(f"T = {T:<8} → P(collision) = {prob:.4f}")