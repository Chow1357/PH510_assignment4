#!/opt/software/anaconda/python-3.10.9/bin/python
"""
2D Ising model simulation

This program implements the classical nearest-neighbour
2D Ising model on an L x L square lattice with
periodic boundary conditions. Metropolis monte carlo 
sampling is used to obtain thermodynamic properties 
including the average energy per site, specific heat
capacity and magnetisation as a function of temperature.

Tested using:
    Python 3.10.9
"""
import numpy as np
from mpi4py import MPI # pylint: disable=no-name-in-module
import matplotlib.pyplot as plt # pylint: disable=no-name-in-module
import matplotlib
matplotlib.use("Agg")

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
    Computing total energy of the lattice using 
    nearest-neighbour interactions
    """

    size = lattice.shape[0]
    energy = 0.0

    for i in range(size):
        for j in range(size):
            s = lattice[i, j]

            # periodic neighbours
            right = lattice[i, (j + 1) % size]
            down = lattice[(i + 1) % size, j]

            # implementing H = 0
            h_field = 0.0
            energy += -j_val * s * (right + down) - h_field * s

    return energy

def delta_energy(spins, i, j, j_val):
    """
    function which finds the value of the system
    upon flipping one spin
    """
    size = spins.shape[0]
    spin = spins[i, j]

    neighbour_sum = (
        spins[(i + 1) % size, j] +
        spins[(i - 1) % size, j] +
        spins[i, (j + 1) % size] +
        spins[i, (j - 1) % size]
    )
    # returning energy chnage for one spin change
    return 2.0 * j_val * spin * neighbour_sum

def magnetisation(lattice):
    """
    Compute the total magnetisation.
    How alligned the entire system is.
    """
    # adding up all the spins to check allignment
    return np.sum(lattice)

#----------------------------------------------
# Task 2 functions for metropolis sampling
#----------------------------------------------
def metropolis_step(lattice, temperature, rng, j_val=1.0):
    """
    Attempt one metropolis spin flip on a randomly chosen lattice site.

    A site is chosen at random. If flipping its spin lowers the energy
    ,the flip is always accepted. If it raises the energy, the flip is 
    accepted with probability exp(-delta_E / kBT),


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
    perform one full Monte Carlo sweep of the lattice.

    Attempts L*L  spin flips, each at a randomly chosen site.
    One sweep is considered one unit of Monte Carlo time.

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

# pylint: disable=too-many-arguments,too-many-positional-arguments
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
    mean_abs_magnetisation = np.mean(np.abs(magnetisation_history))

    # Specific heat per site: Cv/N = 
    # kB = 1 in units of J, N = size^2
    specific_heat = (mean_energy_sq - mean_energy ** 2) / (
        temperature ** 2 * size ** 2
    )

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
    N_SWEEPS = 500
    BURN_IN = 100


    local_seed = 1234 + RANK

    # task 2 tests
    # running simulation
    if RANK == 0:

        print("2D Ising Model - Metropolis Monte Carlo")
        print(f"Lattice size:       {L} x {L}")
        print(f"Temperature range:  {TEMPERATURES[0]:.2f} to "
              f"{TEMPERATURES[-1]:.2f} kBT/J")
        print(f"Temperature points: {len(TEMPERATURES)}")
        print(f"MC sweeps:          {N_SWEEPS}")
        print(f"Burn-in sweeps:     {BURN_IN}")
        print(f"Parallel walkers:   {N_RANKS}")
        print()

        # sanity check: ordered lattice should give energy = -2*J*L^2
        ordered_lattice = initialise_lattice(L, ordered=True)
        expected_energy = -2.0 * J_VAL * L * L
        computed_energy = total_energy(ordered_lattice, J_VAL)
        print("Sanity check (ordered lattice):")
        print(f"  Expected energy: {expected_energy:.1f}")
        print(f"  Computed energy: {computed_energy:.1f}")
        print()

        # ordered_lattice = initialise_lattice(L, ordered=False, seed=1234)
        # total_e = total_energy(spin_lattice, J_VAL)
        # mag = magnetisation(spin_lattice)

        # printing main results
        # print("Random spin lattice:")
        # print(spin_lattice)
        # print()
        # print(f"Total energy: {total_e}")
        # print(f"Magnetisation: {mag}")
        # print(f"Magnetisation per site: {mag / (L * L):.6f}")
        # print(f"Energy per site: {total_e / (L * L):.6f}")
        # print()

    # storage arrays for values from the rnage of temperatures
    temp_results = []
    energy_results = []
    cv_results = []
    mag_results = []

    # each MPI rank runs its own walker for each temperature
    for temp in TEMPERATURES:

        (
            final_lattice,
            sim_energies,
            sim_magnetisations,
            local_mean_energy,
            local_mean_abs_mag,
            local_cv,
        ) = run_simulation(
            size=L,
            temperature=temp,
            n_sweeps=N_SWEEPS,
            j_val=J_VAL,
            seed=local_seed,
            burn_in=BURN_IN
        )
        # these lines send each ranks local averages to rank 0 and then sum them
        total_mean_energy = COMM.reduce(local_mean_energy, op=MPI.SUM, root=0)
   
        total_mean_abs_mag = COMM.reduce(local_mean_abs_mag, op=MPI.SUM, root=0)

        total_cv = COMM.reduce(local_cv, op=MPI.SUM, root=0)

    # computing the global average at rank 0
        if RANK == 0:
            global_mean_energy = total_mean_energy / N_RANKS
            global_cv = total_cv / N_RANKS
            global_mean_abs_mag = total_mean_abs_mag / N_RANKS

            # converting to per-ste quantities
            energy_per_site = global_mean_energy / (L * L)
            mag_per_site = global_mean_abs_mag / (L * L)



            # storing the results for plotting
            temp_results.append(temp)
            energy_results.append(energy_per_site)
            cv_results.append(global_cv)
            mag_results.append(mag_per_site)

            # printing a summary of the key results for each temperature
            print(
                f"T = {temp:.2f}, "
                f"<E>/N = {energy_per_site:.6f}, "
                f"Cv/N = {global_cv:.6f}, "
                f"<|M|>/N = {mag_per_site:.6f}"
            )
    # plots to show temperature dependance of the key parameters
    if RANK == 0:
        # plotting the energy vs temperature
        plt.figure()
        plt.plot(temp_results, energy_results, marker="o")
        plt.xlabel("Temperature (k_B T / J)")
        plt.ylabel("Average energy per site. <E>/N")
        plt.title("2D Ising model: Energy vs Temperature")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ising_energy_vs_temperature.png", dpi=300)
        plt.close()

        # plotting the specific heat per site against temperature
        # should look gaussian
        plt.figure()
        plt.plot(temp_results, cv_results, marker="o")
        plt.xlabel("Temperature (k_B T / J)")
        plt.ylabel("Specific heat per site. V_v/N")
        plt.title("2D Ising model: Specific heat vs Temperature")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ising_cv_vs_temperature.png", dpi=300)
        plt.close()

        #
        plt.figure()
        plt.plot(temp_results, mag_results, marker="o")
        plt.xlabel("Temperature (k_B T / J)")
        plt.ylabel("Average |magnetisation| per site, <|M|>/N")
        plt.title("2D Ising model: Magnetisation vs Temperature")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ising_magnetisation_vs_temperature.png", dpi=300)
        plt.close()

        #
        print("\nSaved plots:")
        print("ising_energy_vs_temperature.png")
        print("ising_cv_vs_temperature.png")
        print("ising_magnetisation_vs_temperature.png")
