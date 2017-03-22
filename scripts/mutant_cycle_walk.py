__doc__ = """
This script produces a walk away from a wildtype lattice sequence.
"""
import os
import argparse
import numpy as np
import itertools as it
import pickle

from gpmap.utils import AMINO_ACIDS
from gpmap.utils import find_differences

from latticeproteins.fitness import Fitness
from latticeproteins.conformations import Conformations
from latticeproteins.sequences import RandomSequence



def calculate_dGs(wildtype):
    """Create a dictionary of first-order and second-order ddGs"""
    length = len(wildtype)

    confs = Conformations(length, database_dir="database")
    fitness = Fitness(1, confs, dGdependence="fracfolded")

    combos = []
    sites = list(range(length))
    dG0 = fitness.AllMetrics(wildtype)[1]

    # Calculate first order coefs
    dG1 = {}
    for i in sites:
        other_sites = sites[:]
        other_sites.remove(i)
        for aa in AMINO_ACIDS:
            combos.append((i, aa))

    for c in combos:
        seq = wildtype[:]
        seq[c[0]] = c[1]
        dG1[c] = fitness.AllMetrics(seq)[1]

    # Calculate second order coefs
    combos = []
    sites = list(range(length))

    for i in sites:
        other_sites = sites[:]
        other_sites.remove(i)
        for aa in AMINO_ACIDS:
            for j in other_sites:
                for aa2 in AMINO_ACIDS:
                    combos.append((i,aa,j,aa2))
    dG2 = {}
    for c in combos:
        seq = wildtype[:]
        seq[c[0]] = c[1]
        seq[c[2]] = c[3]
        # Calculate dG2
        dG2[c] = fitness.AllMetrics(seq)[1] - (dG0 + dG1[(c[0],c[1])] + dG1[(c[2],c[3])])

    # concatenate dGs
    dGs = {}
    dGs.update(dG1)
    dGs.update(dG2)

    return dG0, dGs

def get_coefs(wt, seq):
    """Find mutations and epistatic interactions between a wildtype and sequence"""
    loci = find_differences(wt, seq)
    add = [(pair[0], seq[pair[0]]) for pair in it.combinations(loci, 1)]
    pairs = [(pair[0], seq[pair[0]], pair[1], seq[pair[1]]) for pair in it.combinations(loci, 2)]
    return add, pairs

def predict_dG(wildtype, seq, dG0, dGs):
    """Calculate a predicted stability for a sequence, using dGs from wildtype."""
    add, pairs = get_coefs(wildtype, seq)
    dgs = add + pairs
    stability = float(dG0)
    for coef in dgs:
        stability += dGs[coef]
    return stability

def kmax2d(arr, k):
    """Return the indices of the n largest arguments in the 2d array.
    """
    n, m = arr.shape
    vec = arr.flatten()
    vec_ = vec.argsort()[::-1]
    top_vec = vec_[:k]
    top_n = top_vec // n
    top_m = top_vec % n
    return top_n, top_m

def predicted_open_walks(seq, n_mutations, n_top_mutations, temp=1.0, target=None):
    """
    """
    wildtype = seq
    length = len(wildtype)
    mutant = list(wildtype)
    dG0, dGs = calculate_dGs(wildtype)

    indices = list(range(len(wildtype)))
    paths = [["".join(wildtype)]]

    # Construct a grid of ammino acids at every site
    AA_grid = np.array([AMINO_ACIDS]*length)

    # Start evolving
    for m in range(n_mutations):
        updated_paths = []
        # Iterate through paths
        for i, p in enumerate(paths):
            # construct new trajectories
            new_paths = p * n_top_mutations
            mutant = list(p[-1])

            # Construct grid of all stabilities of all amino acids at all sites
            AA_grid = np.array([AMINO_ACIDS]*length)
            dG = np.zeros(AA_grid.shape, dtype=float)
            for (i,j), AA in np.ndenumerate(AA_grid):
                seq1 = mutant[:]
                seq1[i] = AA_grid[i,j]
                #print(seq, seq1)
                dG[i,j] = predict_dG(seq, seq1, dG0, dGs)

            # Find the top moves in the stability grid
            x, y = kmax2d(dG, n_top_mutations)
            # construct moves from top moves
            new_paths = [p[:] for j in range(n_top_mutations)]

            for k in range(n_top_mutations):
                m = mutant[:]
                m[y[k]] = AA_grid[y[k], x[k]]
                new_paths[k].append("".join(m))

            updated_paths += new_paths
        paths = updated_paths

    return paths


if __name__ == "__main__":

    # Handle command line argument
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ancestor", type=str, help="Ancestral sequence to evolve from.")
    parser.add_argument("n_mutations", type=int, help="number of mutations.")
    args = parser.parse_args()

    # Ancestral sequence to evolve.
    seq = list(args.ancestor)
    out = predicted_open_walks(seq, args.n_mutations, 2)

    # Write to file
    with open("predicted-walks.pickle", "wb") as f:
        pickle.dump(out, f)
