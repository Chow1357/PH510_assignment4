#!/opt/software/anaconda/python-3.10.9/bin/python
"""
2D Ising model simulation

This program implements the classical nearest-neighbour 
2D ISing model on an L x L square lattice with 
periodic boundary conditions.

Tested using:
    Python 3.10.9
"""
import numpy as np 

#size is equivalent to L for grid size
def initialise_lattice(size, ordered=False, seed=1234):
    """
    Create an LxL (sizexsize) Ising spin Lattice.
    """
    rng = np.random.default_rng(seed)

    if ordered:
        return np.ones((size, size), dtype=int)

    return rng.choice([-1, 1], (size, size))


def total_energy(spins, J=1.0)

