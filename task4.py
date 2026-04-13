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
License:
    MIT License

    Copyright (c) 2026 Callum Howie

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files...

    (See full license in the LICENSE file in this repository)
"""
from mpi4py import MPI  # pylint: disable=no-name-in-module
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
N_RANKS = COMM.Get_size()

# step size for metropolis angle pertubation
# pi/4 gives approximatley 50% acceptance rate
DELTA = np.pi / 4

# initialise lattice with angles
def initialise_lattice(size, seed=1234):
    """
    establishing a lattice of size (L x L) populated with
    random angles.

    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, (size, size))

def total_energy(lattice, j_val=1.0):
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
    Compute the change in enrgy from updating one spin angle.
    only the four nearest neighbours contribute to the local energy.
    """
    size = lattice.shape[0]
    old_angle = lattice[row, col]

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

    C(r) = <cos(theta_i - theta_j)> averaged across all pairs separated 
    by distance r along the x-axis. Measures how spin alignment decays
    with distance. Computed up to r = L//2 to avoid periodic image effects.
    """

    size = lattice.shape[0]

    if max_r is None:
        max_r = size // 2

    distances = np.arange(0, max_r + 1)
    correlations = np.zeros(len(distances))

    for idx, r in enumerate(distances):
        # Shift lattice by r along x-axis (axis=1) with periodic wrapping
        shifted = np.roll(lattice, -r, axis=1)
        correlations[idx] = np.mean(np.cos(lattice - shifted))

    return distances, correlations

def count_vortices(lattice):
    """
    Count the vortex density in the XY lattice.

    A vortex is detected by computing the circulation of
    angles around each elementary plaquette. A circulation
    of +2*pi indicates a vortex and -2*pi an anti-vortex.
    Both are counted as they both disrupt long-range order.

    Args:
        lattice (numpy.ndarray): Current spin angle configuration.

    Returns:
        float: Vortex density — number of vortices per lattice site.
    """
    size = lattice.shape[0]
    n_vortices = 0

    for row in range(size):
        for col in range(size):
            # Angle differences around plaquette clockwise
            d1 = lattice[row, (col + 1) % size] - lattice[row, col]
            d2 = (lattice[(row + 1) % size, (col + 1) % size] -
                  lattice[row, (col + 1) % size])
            d3 = (lattice[(row + 1) % size, col] -
                  lattice[(row + 1) % size, (col + 1) % size])
            d4 = lattice[row, col] - lattice[(row + 1) % size, col]

            # Wrap each difference to [-pi, pi]
            d1 = (d1 + np.pi) % (2 * np.pi) - np.pi
            d2 = (d2 + np.pi) % (2 * np.pi) - np.pi
            d3 = (d3 + np.pi) % (2 * np.pi) - np.pi
            d4 = (d4 + np.pi) % (2 * np.pi) - np.pi

            # Total circulation around plaquette
            circulation = d1 + d2 + d3 + d4

            # Vortex or anti-vortex if |circulation| > pi
            if abs(circulation) > np.pi:
                n_vortices += 1

    return n_vortices / (size * size)

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def run_simulation(size, temperature, n_sweeps, j_val=1.0, seed=1234,
                   burn_in=500):
    """
    Run the XY model Metropolis simulation at a fixed temperature.

    Performs n_sweeps Monte Carlo sweeps discarding the first burn_in sweeps
    as equilibration. Computes specific heat from the variance of the energy
    history, spin correlation from the final lattice configuration and
    vortex density from the plaquette circulation. 
    """
    rng = np.random.default_rng(seed)
    lattice = initialise_lattice(size, seed=seed)

    energy_history = []
    total_accepted = 0

    for sweep in range(n_sweeps):
        accepted = monte_carlo_sweep(lattice, temperature, rng, j_val)
        total_accepted += accepted

        if sweep >= burn_in:
            energy_history.append(total_energy(lattice, j_val))

    mean_energy = np.mean(energy_history)
    mean_energy_sq = np.mean(np.square(energy_history))

    specific_heat = (mean_energy_sq - mean_energy ** 2) / (
        temperature ** 2 * size ** 2
    )

    acceptance_rate = total_accepted / (n_sweeps * size * size)

    distances, correlations = spin_correlation(lattice)
    vortex_density = count_vortices(lattice)

    return (
        mean_energy,
        specific_heat,
        distances,
        correlations,
        acceptance_rate,
        vortex_density,
    )

if __name__ == "__main__":
    # simulation parameters
    J_VAL = 1.0
    # Use finer spacing near Tc
    TEMPERATURES = np.concatenate([
        np.linspace(0.5, 0.75, 5),    # coarse below transition
        np.linspace(0.75, 1.25, 21)[1:],   # fine near peak
        np.linspace(1.25, 1.5, 3)[1:]     # coarse above transition
    ])
    N_SWEEPS = 10000
    BURN_IN = 1000

    # lattice sizes to simulate for finite size scaling
    LATTICE_SIZES = [16, 32, 64]

    # Temperature to examine correlation behaviour across sizes
    CORRELATION_TEMP = 1.0

    local_seed = 1234 + RANK

    if RANK == 0:
        print("2D XY Model - Metropolis Monte Carlo")
        print(f"Lattice size:       {LATTICE_SIZES}")
        print(f"Temperature range:  {TEMPERATURES[0]:.2f} to "
              f"{TEMPERATURES[-1]:.2f} kBT/J")
        print(f"Temperature points: {len(TEMPERATURES)}")
        print(f"MC sweeps:          {N_SWEEPS}")
        print(f"Burn-in sweeps:     {BURN_IN}")
        print(f"Parallel walkers:   {N_RANKS}")
        print(f"Angle step size:    {DELTA:.4f} rad ({np.degrees(DELTA):.1f} deg)")
        print()

    # Storage dictionaries keyed by lattice size
    all_cv_results = {}
    all_temp_results = {}
    all_energy_results = {}
    # Correlation arrays stored per size and temperature
    all_correlations = {}
    all_vortex_results = {}

    for lattice_size in LATTICE_SIZES:

        if RANK == 0:
            print(f"--- Simulating L = {lattice_size} ---")
        # Storage for results (only populated on rank 0)
        temp_results = []
        energy_results = []
        cv_results = []
        # Store full correlation arrays for a selection of temperatures
        correlation_data = {}
        vortex_results = []

        for temp in TEMPERATURES:

            (
                local_mean_energy,
                local_cv,
                r_values,
                local_correlations,
                local_acceptance,
                local_vortex_density,
            ) = run_simulation(
                size=lattice_size,
                temperature=temp,
                n_sweeps=N_SWEEPS,
                j_val=J_VAL,
                seed=local_seed,
                burn_in=BURN_IN,
            )

            # Reduce scalar quantities to rank 0
            total_mean_energy = COMM.reduce(
                local_mean_energy, op=MPI.SUM, root=0
            )
            total_cv = COMM.reduce(local_cv, op=MPI.SUM, root=0)
            total_acceptance = COMM.reduce(
                local_acceptance, op=MPI.SUM, root=0
            )

            # Reduce correlation array element-wise to rank 0
            total_correlations = COMM.reduce(
                local_correlations, op=MPI.SUM, root=0
            )
            # MPI reduction for the vortex density calculations
            total_vortex_density = COMM.reduce(
                local_vortex_density, op=MPI.SUM, root=0
            )

            if RANK == 0:
                # global averages across the lattice
                global_mean_energy = total_mean_energy / N_RANKS
                global_cv = total_cv / N_RANKS
                global_acceptance = total_acceptance / N_RANKS
                global_correlations = total_correlations / N_RANKS
                global_vortex_density = total_vortex_density / N_RANKS

                energy_per_site = global_mean_energy / (lattice_size * lattice_size)

                temp_results.append(temp)
                cv_results.append(global_cv)
                energy_results.append(energy_per_site)
                correlation_data[round(temp, 2)] = global_correlations
                vortex_results.append(global_vortex_density)

                print(
                    f"L = {lattice_size}, T = {temp:.2f} | "
                    f"<E>/N = {energy_per_site:.4f} | "
                    f"Cv/N = {global_cv:.4f} | "
                    f"Acceptance = {global_acceptance:.2%} | "
                    f"Vortex density = {global_vortex_density:.4f}"
                )

        # Store all results for this lattice size
        if RANK == 0:
            all_cv_results[lattice_size] = cv_results
            all_temp_results[lattice_size] = temp_results
            all_energy_results[lattice_size] = energy_results
            all_correlations[lattice_size] = correlation_data
            all_vortex_results[lattice_size] = vortex_results
            print()

    if RANK == 0:
        colours = ["steelblue", "darkorange", "seagreen", "crimson"]

        # plot: specific Heat vs temperature
        plt.figure()
        for lattice_size, colour in zip(LATTICE_SIZES, colours):
            plt.plot(
                all_temp_results[lattice_size],
                all_cv_results[lattice_size],
                marker="o",
                label=f"L = {lattice_size}",
                color=colour,
            )
        plt.xlabel("Temperature ($k_B T / J$)")
        plt.ylabel(r"Specific heat per site $C_v / N$")
        plt.title("2D XY Model: Specific Heat vs Temperature")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("xy_cv_vs_temperature.png", dpi=300)
        plt.close()

        # plot: Spin Correlation vs Fractionl distance
        # plotting for a selection of temperaturs to show
        # change in behaviour
        plt.figure()
        for lattice_size, colour in zip(LATTICE_SIZES, colours):
            corr = all_correlations[lattice_size].get(
                round(CORRELATION_TEMP, 2), None
            )
            if corr is not None:
                # Fractional distance r/L is consistent across sizes
                r_over_l = np.arange(len(corr)) / lattice_size
                plt.plot(
                    r_over_l,
                    corr,
                    marker="o",
                    label=f"L = {lattice_size}",
                    color=colour,
                )
        plt.xlabel("Fractional lattice distance $r/L$")
        plt.ylabel(r"Spin correlation $C(r)$")
        plt.title(
            f"2D XY Model: Spin Correlation at T = {CORRELATION_TEMP:.2f} "
            "for Multiple Lattice Sizes"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("xy_correlation_finite_size.png", dpi=300)
        plt.close()

        # plot: Energy vs Temperature
        plt.figure()
        for lattice_size, colour in zip(LATTICE_SIZES, colours):
            plt.plot(
                all_temp_results[lattice_size],
                all_energy_results[lattice_size],
                marker="o",
                label=f"L = {lattice_size}",
                color=colour,
            )
        plt.xlabel("Temperature ($k_B T / J$)")
        plt.ylabel(r"Average energy per site $\langle E \rangle / N$")
        plt.title(
            "2D XY Model: Finite Size Scaling of Energy"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("xy_energy_finite_size.png", dpi=300)
        plt.close()

        # Plot: Spin correlation vs fractional distance
        # at multiple temperatures for L=16
        # Shows change from power-law to exponential decay
        # across the BKT transition
        PLOT_TEMPS = [0.5, 0.75, 1.0, 1.25, 1.5]
        temp_colours = ["navy", "steelblue", "seagreen",
                        "darkorange", "crimson"]

        plt.figure()
        TARGET_SIZE = LATTICE_SIZES[0]

        for plot_temp, temp_colour in zip(PLOT_TEMPS, temp_colours):
            key = round(plot_temp, 2)
            corr = all_correlations[TARGET_SIZE].get(key, None)
            if corr is not None:
                r_over_l = np.arange(len(corr)) / TARGET_SIZE
                plt.plot(
                    r_over_l,
                    corr,
                    marker="o",
                    label=f"T = {plot_temp:.2f}",
                    color=temp_colour,
                )

        plt.xlabel("Fractional lattice distance $r/L$")
        plt.ylabel(
            r"Spin correlation $C(r) = "
            r"\langle \cos(\theta_i - \theta_j) \rangle$"
        )
        plt.title("2D XY Model: Spin Correlation vs Distance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("xy_spin_correlation_temps.png", dpi=300)
        plt.close()

        # Plot: Vortex density vs temperature
        # Low density below Tc increases sharply above Tc
        # consistent with BKT vortex pair unbinding
        plt.figure()
        for lattice_size, colour in zip(LATTICE_SIZES, colours):
            plt.plot(
                all_temp_results[lattice_size],
                all_vortex_results[lattice_size],
                marker="o",
                label=f"L = {lattice_size}",
                color=colour,
            )
        plt.xlabel("Temperature ($k_B T / J$)")
        plt.ylabel("Vortex density (vortices per site)")
        plt.title("2D XY Model: Vortex Density vs Temperature")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("xy_vortex_density.png", dpi=300)
        plt.close()

        print("Plots saved:")
        print("  xy_cv_vs_temperature.png")
        print("  xy_correlation_finite_size.png")
        print("  xy_energy_finite_size.png")
        print("  xy_spin_correlation_temps.png")
        print("  xy_vortex_density.png")
