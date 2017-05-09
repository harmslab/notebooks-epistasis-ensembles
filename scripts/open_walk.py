__doc__ = """
This script produces a walk away from a wildtype lattice sequence.
"""
import pickle
import numpy as np
import argparse

from latticeproteins.fitness import Fitness
from latticeproteins.interactions import miyazawa_jernigan
from latticeproteins.sequences import RandomSequence, NMutants
from latticeproteins.conformations import Conformations

from gpmap.utils import hamming_distance
from gpmap.utils import AMINO_ACIDS

from latticegpm.gpm import LatticeGenotypePhenotypeMap


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

def open_walks(seq, n_mutations, n_top_mutations, temp=1.0, target=None):
    """
    """
    length = len(seq)
    c = Conformations(length, "database")
    dGdependence = "fracfolded"

    wildtype = seq
    mutant = list(wildtype)

    indices = list(range(len(wildtype)))
    paths = [[wildtype]]

    # Get native structure
    fitness = Fitness(temp, c, dGdependence=dGdependence)
    wt_structure = fitness._NativeE(wildtype)[1]

    # Construct a grid of ammino acids at every site
    AA_grid = np.array([AMINO_ACIDS]*length)

    # Start evolving
    for m in range(n_mutations):
        updated_paths = []
        # Iterate through paths
        for l, p in enumerate(paths):
            # construct new trajectories
            new_paths = p * n_top_mutations
            mutant = list(p[-1])

            # Construct grid of all stabilities of all amino acids at all sites
            AA_grid = np.array([AMINO_ACIDS]*length)
            dG = np.zeros(AA_grid.shape, dtype=float)
            for (i,j), AA in np.ndenumerate(AA_grid):
                seq1 = mutant[:]
                seq1[i] = AA_grid[i,j]
                dG[i,j] = fitness.Stability(seq1)

            # Find the top moves in the stability grid
            x, y = kmax2d(dG, n_top_mutations)
            # construct moves from top moves
            new_paths = [p[:] for q in range(n_top_mutations)]

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
    out = open_walks(seq, args.n_mutations, 10)

    # Write to file
    with open("results/actual-walks.pickle", "wb") as f:
        pickle.dump(out, f)