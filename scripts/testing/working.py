__doc__ = """
The point of this simulation is to observe how population size effects our predictions
of evolutionary trajectories.

In the non-infinite regime, all evolutionary trajectories are technically possible.
Since we cannot enumerate all possible trajectoreis (as this grows wuite quickly)
we limit ourselves to a set of sampled trajectories and observe how these change
with population size.

We
"""


import os, sys
import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
import pickle
import argparse
from multiprocessing import Process

# Latticeprotein imports
from latticeproteins.thermodynamics import LatticeThermodynamics
from latticeproteins.interactions import miyazawa_jernigan
from latticeproteins.conformations import ConformationList, Conformations
from latticeproteins.sequences import find_differences, _residues
from latticeproteins.evolve import monte_carlo_fixation_walk, fixation
from latticeproteins.sequences import random_sequence

POPSIZE = 1

def fitness_matrix(sequence, lattice, trait="fracfolded"):
    """Calculate a fitness matrix for a given lattice sequence. This function
    enumerates all possible single mutations


    Parameters
    ----------
    sequence : str
        sequence to mutate
    lattice : LatticeThermodynamics object or subobject
        Lattice object to calculate fitness
    trait : str
        attribute to use for fitness calculator.

    Returns
    -------
    F = Fitness matrix
    """
    length = len(sequences)
    F = pd.Dataframe(index=_residues, columns=list(sequence))
    
    
    
    
    
    # Construct grid of the stabilities for all neighbors in sequence space (1 step away.)
    AA_grid = np.array([_residues]*length) # enumerate residues at all sites
    F = np.zeros(AA_grid.shape, dtype=float)
    for (i,j), AA in np.ndenumerate(AA_grid):
        seq1 = sequence[:]
        seq1[i] = AA_grid[i,j]
        fits[i,j] = fitness_method(seq1, target=target)



    return F




def trajectories_true(anc, temp, target, confs, db, dir):
    """"""
    lattice = LatticeThermodynamics(temp, confs)





if __name__ == "__main__":
    # Handle command line argument
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ancestor", type=str, help="Ancestral sequence to evolve from.")
    parser.add_argument("n_mut", type=int, help="number of mutations.")
    parser.add_argument("i", type=str, help="dataset number (arbitrary).")
    parser.add_argument("n_states", type=int, help="Number of states in ensemble. 0 means all states.")
    parser.add_argument("--conf", default=None, help="target structure.", type=str)
    parser.add_argument("--db", nargs=3, help="target structure.", type=str)
    parser.add_argument("--temp", default=1, help="temperature", type=float)
    args = parser.parse_args()

    temp = args.temp
    seq = args.ancestor
    target = args.conf
    db = args.db

    # Construct a conformations database
    length = len(seq)
    if args.n_states == 0:
        confs = Conformations(length, "database")
        directory = "full-state"
    else:
        directory = "{}-state".format(args.n_states)
        c = db[0:args.n_states]
        confs = ConformationList(length, c)

    # Randomly walk in the true space.
