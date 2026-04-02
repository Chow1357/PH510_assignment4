#!/opt/software/anaconda/python-3.10.9/bin/python
"""
2D XY model simulation
 
This program implements the classical nearest-neighbour
2D XY model on an L x L square lattice with periodic
boundary conditions. Each spin is a continuous angle
theta in [0, 2*pi). Metropolis Monte Carlo sampling is
used to obtain thermodynamic properties including the
specific heat capacity and spin correlation function
as a function of temperature.
 
Tested using:
    Python 3.10.9
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI  # pylint: disable=no-name-in-module
 
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
N_RANKS = COMM.Get_size()

# step size for metropolis angle pertubation 
# pi/4 gives approximatley 50% acceptance rate
DELTA = np.pi / 4

# initialise lattice with angles 
def initialise_lattice(size, ordered=False, seed=1234)
    """
    establishing a lattice of size (L x L) populated with 
    different angles

    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, (size, size))

def total_energy(lattice, j_val=1.0)
    """
    compute the total energy of the XY lattice
    """

    # np.roll shifts the array by 1 position with periodic wrapping,
    # giving us the neighbour angles at every site simultaneously.
    right_neighbours = np.roll(lattice, -1, axis=1)
    down_neighbours = np.roll(lattice, -1, axis=0)
 
    energy = -j_val * np.sum(
        np.cos(lattice - right_neighbours) +
        np.cos(lattice - down_neighbours)
    )
 
    return energy

def delta_energy(lattice, row, col, new_angle, j_val=1.0):
    """
    Compute the chnage in enrgy from updating one spin angle. 
    only the four nearest neighbours contribute to the local energy.
    """
    size = lattice.shape[0]
    old_angle = lattice[row, col]
 
    # Sum contributions from all four nearest neighbours
    neighbour_angles = (
        lattice[(row + 1) % size, col] +
        lattice[(row - 1) % size, col] +
        lattice[row, (col + 1) % size] +
        lattice[row, (col - 1) % size]
    )
 
    # Energy change = E_new - E_old for this site only
    # Each neighbour contributes -J*cos(theta_i - theta_neighbour)
    old_interaction = np.cos(old_angle - lattice[(row + 1) % size, col])
    old_interaction += np.cos(old_angle - lattice[(row - 1) % size, col])
    old_interaction += np.cos(old_angle - lattice[row, (col + 1) % size])
    old_interaction += np.cos(old_angle - lattice[row, (col - 1) % size])
 
    new_interaction = np.cos(new_angle - lattice[(row + 1) % size, col])
    new_interaction += np.cos(new_angle - lattice[(row - 1) % size, col])
    new_interaction += np.cos(new_angle - lattice[row, (col + 1) % size])
    new_interaction += np.cos(new_angle - lattice[row, (col - 1) % size])
 
    # Unused variable removed; neighbour_angles used implicitly above
    _ = neighbour_angles
 
    return -j_val * (new_interaction - old_interaction)

def metropolis_step(lattice, temperature, rng, j_val=1.0):
    """
    Attempt one metropolis update on a randomly chosen site.
    """
    size = lattice.shape[0]
 
    row = rng.integers(0, size)
    col = rng.integers(0, size)
 
    # Perturb angle by a random amount in [-DELTA, +DELTA]
    # and wrap to keep it within [0, 2*pi)
    perturbation = rng.uniform(-DELTA, DELTA)
    new_angle = (lattice[row, col] + perturbation) % (2 * np.pi)
 
    d_e = delta_energy(lattice, row, col, new_angle, j_val)

    if d_e <= 0:
        lattice[row, col] = new_angle
        return True
 
    if rng.random() < np.exp(-d_e / temperature):
        lattice[row, col] = new_angle
        return True
 
    return False

def monte_carlo_sweep(lattice, temperature, rng, j_val=1.0):
    """
    perform one full monte carlo sweep of the XY lattice.

    Attempts L*L angle updates, each at a randomly chosen site.
    One sweep is considered one unit of Monte Carlo time.
    """
    size = lattice.shape[0]
    accepted_moves = 0
 
    for _ in range(size * size):
        if metropolis_step(lattice, temperature, rng, j_val):
            accepted_moves += 1
 
    return accepted_moves

def spin_correlation(lattice, max_r=None):
    """
    Compute the spin correlation function C(r) as a function of distance.

    C(r) = 
    """

    size = lattice.shape[0]
 
    if max_r is None:
        max_r = size // 2
 
    r_values = np.arange(0, max_r + 1)
    correlations = np.zeros(len(r_values))
 
    for idx, r in enumerate(r_values):
        # Shift lattice by r along x-axis (axis=1) with periodic wrapping
        shifted = np.roll(lattice, -r, axis=1)
        correlations[idx] = np.mean(np.cos(lattice - shifted))
 
    return r_values, correlations
