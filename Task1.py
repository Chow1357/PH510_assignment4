#!/opt/software/anaconda/python-3.10.9/bin/python
"""
2D Ising model simulation

This program implements the classical nearest-neighbour 2D ISing model on an L x L
square lattice with periodic boundary conditions.

Tested using:
    Python 3.10.9
"""
import numpy as np 

def initialise_lattice(L, ordered=False, seed=None)
