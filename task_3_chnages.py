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
from mpi4py import MPI # pylint: disable=no-name-in-module

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
N_RANKS = COMM.Get_size()

#size is equivalent to L for grid size
def initialise_lattice(size, ordered=False, seed=1234):
    """
    Create an LxL (sizexsize) Ising spin Lattice.
    """
    rng = np.random.default_rng(seed)

    if ordered:
        return np.ones((size, size), dtype=int)

    return rng.choice([-1, 1], (size, size))


def total_energy(lattice, j_val=1.0):
    """
    Computing total energy of the lattice using nearest-neighbour interactions
    """

    l = lattice.shape[0]
    energy = 0.0

    for i in range(l):
        for j in range(l):
            s = lattice[i, j]

            # periodic neighbours
            right = lattice[i, (j + 1) % l]
            down = lattice[(i + 1) % l, j]

            # implementing H = 0
            h_field = 0.0
            energy += -j_val * s * (right + down) - h_field * s

    return energy

def delta_energy(spins, i, j, j_val):
    """
    function which finds the value of the system
    upon flipping one spin
    """
    l = spins.shape[0]
    s = spins[i, j]

    neighbour_sum = (
        spins[(i + 1) % l, j] +
        spins[(i - 1) % l, j] +
        spins[i, (j + 1) % l] +
        spins[i, (j - 1) % l]
    )
    # returning energy chnage for one spin change
    return 2.0 * j_val * s * neighbour_sum

def magnetisation(spin):
    """
    Compute the total magnetisation.
    How alligned the entire system is.
    """
    # adding up all the spins to check allignment
    return np.sum(spin)

#----------------------------------------------
# Task 2 functions for metropolis sampling
#----------------------------------------------
def metropolis_step(lattice, temperature, rng, j_val=1.0):
    """
    Attempt one metropolis spin flip.


    """
    # gets the number of rows in the lattice
    size = lattice.shape[0]

    # choose a random spin site
    i = rng.integers(0, size)
    j = rng.integers(0, size)

    # energy change from flipping the chosen spin site
    d_e = delta_energy(lattice, i, j, j_val)

    # accept a spin flip that lowers or
    # leaves energy unchnaged
    if d_e <= 0:
        lattice[i, j] *= -1
        return True

    # sometimes accept energy increase flips
    # based off random number compared to
    # acceptance probability
    if rng.random() < np.exp(-d_e / temperature):
        lattice[i, j] *= -1
        return True

    return False
# performing a full sweep through monte carlo time
def monte_carlo_sweep(lattice, temperature, rng, j_val=1.0):
    """
    djdjdkd
    """
    size = lattice.shape[0]
    # counts the number of accepts
    # can be used ot find acceptance rate
    accepted_moves = 0

    # adding accepted moves to counter
    for _ in range(size * size):
        if metropolis_step(lattice, temperature, rng, j_val):
            accepted_moves += 1

    return accepted_moves

def run_simulation(size, temperature, n_sweeps, j_val=1.0, seed=1234,
                   burn_in=20):
    """
    simulation which includes a loop for a selected number of sweeps
    """
    rng = np.random.default_rng(seed)
    # sets the starting configuration of the lattice
    lattice = initialise_lattice(size, ordered=False, seed=seed)

    # arrays to store variables
    energy_history = []
    magnetisation_history = []

    for sweep in range(n_sweeps):
        monte_carlo_sweep(lattice, temperature, rng, j_val)

        if sweep >= burn_in:
            energy_history.append(total_energy(lattice, j_val))
            magnetisation_history.append(magnetisation(lattice))

    mean_energy = np.mean(energy_history)
    mean_energy_sq = np.mean(np.square(energy_history))
    mean_abs_magnetisation = np.mean(np.abs(magnetisation(lattice)))

    return(
        lattice,
        energy_history,
        magnetisation_history,
        mean_energy,
        mean_energy_sq,
        mean_abs_magnetisation
    )

# ensuring this part will only run
# if this file is executed directly
if __name__ == "__main__":
    # setting parameters
    L = 8
    J_VAL = 1.0
    TEMPERATURES = np.linspace(1.0, 3.0, 11)
    N_SWEEPS = 1000
    BURN_IN = 200

    # task 2 tests
    # running simulation
    if RANK == 0:
        # creating a random lattice 
        spin_lattice = initialise_lattice(L, ordered=False, seed=1234)
        total_e = total_energy(spin_lattice, J_VAL)
        mag = magnetisation(spin_lattice)

        # printing main results
        print("Random spin lattice:")
        print(spin_lattice)
        print()
        print(f"Total energy: {total_e}")
        print(f"Magnetisation: {mag}")
        print(f"Magnetisation per site: {mag / (L * L):.6f}")
        print(f"Energy per site: {total_e / (L * L):.6f}")
        print()

        # ordered-lattice test
        ordered_spins = initialise_lattice(L, ordered=True)
        print("Ordered lattice test:")
        print(f"Total energy: {total_energy(ordered_spins, J_VAL)}")
        print(f"Magnetisation: {magnetisation(ordered_spins)}")
        print()

    # each MPI process is assigned a different rank
    # each rank runs one lattice, one metropolis
    # using its own seed at same temp
    local_seed = 1234 + RANK

    # storage arrays for values from the rnage of temperatures
    temp_results = []
    energy_results = []
    cv_results = []
    mag_results = []
    # each MPI rank runs its own walker for each temperature
    # and returns the following data
    for temperatures in TEMPERATURES:

        (
            final_lattice,
            sim_energies,
            sim_magnetisations,
            local_mean_energy,
            local_mean_energy_sq,
            local_mean_abs_magnetisation,
        ) = run_simulation(
            size=L,
            temperature=TEMPERATURE,
            n_sweeps=N_SWEEPS,
            j_val=J_VAL,
            seed=local_seed,
            burn_in=BURN_IN
        )
        # these lines send each ranks local averages to rank 0 and then sum them 
        total_mean_energy = COMM.reduce(local_mean_energy, op=MPI.SUM, root=0)
        total_mean_energy_sq = COMM.reduce(local_mean_energy_sq, op=MPI.SUM, root=0)
        total_mean_abs_mag = COMM.reduce(local_mean_abs_magnetisation, op=MPI.SUM, root=0)

    # computing the global average at rank 0
        if RANK == 0:
            global_mean_energy = total_mean_energy / N_RANKS
            global_mean_energy_sq = total_mean_energy_sq / N_RANKS
            global_mean_abs_mag_per_site = total_mean_abs_mag / N_RANKS
        
            # converting to per-ste quantities
            energy_per_site = global_mean_energy / (L * L)
            mag_per_site = global_mean_abs_mag / (L * L)

            # compute specific heat capacity
            cv = (
                global_mean_energy_sq - global_mean_energy**2
            ) / (temperature**2)
            cv_per_site = cv / (L * L)

            # storing the results for plotting
            temp_results.append(temperature)
            energy_results.append(energy_per_site)
            cv_results.append(cv_per_site)
            mag_results.append(mag_per_site)
 
            # printing a summary of the key results for each temperature
            print(
                f"T = {temperature:.2f}, "
                f"<E>/N = {energy_per_site:.6f}, "
                f"Cv/N = {cv_per_site:.6f}, "
                f"<|M|>/N = {mag_per_site:.6f}"
            )
