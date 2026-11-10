"""多echo对的通用自动微分线路。"""
from __future__ import annotations

import autograd.numpy as np
from autograd import value_and_grad


def evolve(angles, energy, flips):
    """从均匀态演化任意深度线路，紧凑角度排列为gamma后接beta。"""
    depth = len(angles)//2
    state = np.ones(len(energy), dtype=complex)/np.sqrt(len(energy))
    for layer in range(depth):
        state = np.exp(-1j*angles[layer]*energy)*state
        for flip in flips:
            state = np.cos(angles[depth+layer])*state-1j*np.sin(angles[depth+layer])*state[flip]
    return state


def probability(angles, energy, flips, target):
    """统一实值目标概率，可由autograd对任意深度直接求导。"""
    amplitude = evolve(angles, energy, flips)[target]
    return np.real(amplitude*np.conj(amplitude))


probability_gradient = value_and_grad(probability)
