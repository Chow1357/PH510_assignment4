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

            energy += -j_val * s * (right + down)

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

# ensuring this part will only run
# if this file is executed directly
if __name__ == "__main__":
    # setting parameters
    L = 8
    J_VAL = 1.0

    # create a random lattice with a seed so reproducible
    spin_lattice = initialise_lattice(L, ordered=False, seed=1234)

    # calling functions for magentisation and energy
    total_e = total_energy(spin_lattice, J_VAL)
    mag = magnetisation(spin_lattice)

    # printing results
    print("Random spin lattice:")
    print(spin_lattice)
    print()
    print(f"Total energy: {total_e}")
    print(f"Magnetisation: {mag}")
    print(f"Magnetisation per site: {mag / (L * L):.6f}")
    print(f"Energy per site; {total_e / (L * L):.6f}")

    ordered_spins = initialise_lattice(L, ordered=True)
    print("Ordered lattice test:")
    print(f"Total energy: {total_energy(ordered_spins, J_VAL)}")
    print(f"Magnetisation: {magnetisation(ordered_spins)}")
