#!/opt/software/anaconda/python-3.10.9/bin/python
"""
2D Ising model simulation

This program implements the classical nearest-neighbour
2D Ising model on an L x L square lattice with
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


def total_energy(spins, j_val=1.0):
    """
    Computing total energy of the lattice using nearest-neighbour interactions
    """

    g_size = spins.shape[0]
    energy = 0.0

    for i in range(L):
        for j in range(L):
            s = lattice[i, j]

            # periodic neighbours
            right = lattice[i, (j + 1) % L]
            down = lattice[(i + 1) % L, j]

            energy += -j_val * s * (right + down)

    return energy

def delta_energy(spins, i, j, j_val=1.0):
    """
    function which finds the value of the system 
    upon flipping one spin
    """
    L = spins.shape(0)
    S = spins[i, j]

    neighbour_sum = (
        spins[(i + 1) % L, j] +
        spins[(i - 1) % L, j] +
        spins[i, (j + 1) % L] +
        spins[i, (j - 1) % L]
    )
    # returning energy chnage for one spin change
    return 2.0 * j_val * s * neighbour_sum

def magnetisation(spins):
    """
    Compute the total magnetisation.
    How alligned the entire system is.
    """
    # adding up all the spins to check allignment
    return np.sum(spins)

# ensuring this part will only run
# if this file is executed directly
if __name__ == "__main__":
    # setting parameters
    L = 8
    j_val = 1.0

    spins = initialise_lattice(L, ordered=False, seed=1234)
