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
