import os, sys
import numpy as np
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

def trajectories_true(seq, temp, target, confs, db, dir):
    """Enumerate trajectories away from an ancestral lattice sequence."""
    # Create a lattice protein calculator with given temperature and conf database.
    lattice = LatticeThermodynamics(temp, confs)
    edges = enumerate_walks(seq, lattice, target=target, max_mutations=7)
    # Write to file
    out = {"seq":seq, "temp":temp, "target": target, "edges": edges, "db": db}
    with open("results-{}/trajectories-true-{}.pickle".format(dir, args.i), "wb") as f:
        pickle.dump(out, f)

def trajectories_additive(seq, temp, target, confs, db, dir):
    """Predict walks using additive mutational effects"""
    # Create a predicted lattice model
    plattice1 = PredictedLattice(seq, temp, confs, double=False, target=target)
    edges = enumerate_walks(seq, plattice1, target=target, max_mutations=7)
    # Write to file
    out = {"seq":seq, "temp":temp, "target": target, "edges": edges, "db": db}
    with open("results-{}/trajectories-additive-{}.pickle".format(dir, args.i), "wb") as f:
        pickle.dump(out, f)

def trajectories_pairwise(seq, temp, target, confs, db, dir):
    """Predict walks using additive and pairwise mutational effects"""
    # Create a predicted lattice model
    plattice = PredictedLattice(seq, temp, confs, double=True, target=target)
    edges = enumerate_walks(seq, plattice, target=target, max_mutations=7)
    # Write to file
    out = {"seq":seq, "temp":temp, "target": target, "edges": edges, "db": db}
    with open("results-{}/trajectories-pairwise-{}.pickle".format(dir, args.i), "wb") as f:
        pickle.dump(out, f)

def fixation(fitness1, fitness2, *args, **kwargs):
    """ Simple Gillespie fixation probability between two organism with fitnesses 1 and 2.
    (With infinite population size!)

    .. math::
        p_{\\text{fixation}} = \\frac{1 - e^{-N \\frac{f_2-f_1}{f1}}}{1 - e^{-\\frac{f_2-f_1}{f1}}}
    """
    sij = (fitness2 - fitness1)/abs(fitness1)
    # Check if any nans exist if an array of fitnesses is given.
    popsize = 1000
    fixation = (1 - np.exp(-popsize*sij)) / (1 - np.exp(-sij))
    if type(fixation) == np.ndarray:
        fixation = np.nan_to_num(fixation)
        fixation[sij < 0] = 0
    return  fixation

class PredictedLattice(object):
    """Infers the stability and fraction folded of any sequence with respect
    to some wildtype lattice model using a linear, biochemical epistasis model.
    Calculates the independent effect of all mutations (and pairwise effects
    if `double` is True) and sums those effects to predict other sequences.

    Parameters
    ----------
    wildtype : str
        wildtype/ancestral sequence
    temp : float
        temperature in reduced units
    confs : list
        list of conformations
    double : bool
        if True, include pairwise epistasis when inferring stability.
    """
    def __init__(self, wildtype, temp, confs, double=False, target=None):
        self.wildtype = wildtype
        self.temp = temp
        self.conformations = confs
        self.target = target
        self._lattice = LatticeThermodynamics(self.temp, self.conformations)
        self.double = double

        combos = []
        sites = list(range(self.conformations.length()))
        self.dG0 = self._lattice.stability(self.wildtype, target=self.target)

        #####  Build a dictionary of additive and pairwise mutational effects ####
        # Calculate first order coefs
        self.dGs = {}
        for i in sites:
            other_sites = sites[:]
            other_sites.remove(i)
            for aa in _residues:
                combos.append((i, aa))

        for c in combos:
            seq = list(self.wildtype[:])
            seq[c[0]] = c[1]
            # Calculate dG as dG_wt -
            self.dGs[c] = self._lattice.stability(seq, target=self.target) - self.dG0

        if self.double:
            # Calculate second order coefs
            combos = []
            sites = list(range(self.conformations.length()))
            for i in sites:
                other_sites = sites[:]
                other_sites.remove(i)
                for aa in _residues:
                    for j in other_sites:
                        for aa2 in _residues:
                            combos.append((i,aa,j,aa2))

            for c in combos:
                seq = list(self.wildtype[:])
                seq[c[0]] = c[1]
                seq[c[2]] = c[3]
                # Calculate dG2
                self.dGs[c] = self._lattice.stability(seq, target=self.target) - (self.dG0 + self.dGs[(c[0],c[1])]+ self.dGs[(c[2],c[3])])

    def stability(self, seq, target=None):
        """Calculate the stability of a given sequence using the Lattice predictor"""
        # Get additive coefs to build predictions
        if target != self.target:
            raise Exception("Target does not match wildtype target.")
        loci = find_differences(self.wildtype, seq)
        # Get all additive combinations for the sequence given
        add = [(pair[0], seq[pair[0]]) for pair in it.combinations(loci, 1)]
        if self.double:
            # Get all pairwise effects for the sequence given
            pairs = [(pair[0], seq[pair[0]], pair[1], seq[pair[1]]) for pair in it.combinations(loci, 2)]
            dgs = add + pairs
        else:
            dgs = add
        # Get the wildtype stability
        stability = float(self.dG0)
        # Sum the mutational effects
        for coef in dgs:
            stability += self.dGs[coef]
        return stability

    def fracfolded(self, seq, target=None):
        """Calculate the fraction folded for a given sequence"""
        return 1.0 / (1.0 + np.exp(self.stability(seq, target=target) / self.temp))


def enumerate_walks(seq, lattice, selected_trait="fracfolded", max_mutations=5, target=None):
    """Enumerate all probable walks away from an ancestral lattice sequence. The probability
    of a trajectory depends on the selected trait.

    Parameters
    ----------
    seq : str
        seq
    lattice : LatticeThermodynamics object
        Lattice protein calculator
    selected_trait : str
        The trait to select.
    max_mutations : int (default = 15)
        Max number of mutations to make in the walk.
    target : str
        selected lattice target conformation. If None, the lattice will
        fold to the natural native conformation.

    Returns
    -------
    edges : list
        A list of all edges by the random walks out. Each element in the lists
        is a tuple. Tuple[0] is (seq_i, seq_j), Tuple[1] is {"weight" : <fixation probability>}.
    """
    length = len(seq)
    # Get the fitness attribute from the lattice model object
    fitness_method = getattr(lattice, selected_trait)

    # Calculate the wildtype fitness
    fitness0 = fitness_method(seq, target=target)
    finished = False

    # Steps
    moves = [seq]

    # Track fitnesses as we walk
    fitnesses = [fitness0]

    # Inialize a list of edges visited
    edges = []

    # Current mutation
    ith_mutation = 0
    while len(moves) != 0 and ith_mutation < max_mutations:
        #
        new_moves, new_fitnesses = [], []
        for i, m in enumerate(moves):
            # Copy sequence and fitness for storing
            sequence = list(m[:])
            fitness0 = fitnesses[i]

            # Construct grid of the stabilities for all neighbors in sequence space (1 step away.)
            AA_grid = np.array([_residues]*length) # enumerate residues at all sites
            fits = np.zeros(AA_grid.shape, dtype=float)
            for (i,j), AA in np.ndenumerate(AA_grid):
                seq1 = sequence[:]
                seq1[i] = AA_grid[i,j]
                fits[i,j] = fitness_method(seq1, target=target)

            # Calculate fitness for all neighbors in sequence space
            fix = fixation(fitness0, fits)*(1./fits.size) # multiplied by flat prior for all mutations

            # Find all neighbors with a reasonable probability of transition
            site, aa_index = np.where(fix > 1.0e-9)
            AA = AA_grid[site, aa_index]
            FF = fits[site, aa_index]
            prob = fix[site, aa_index]

            # Iterate over viable neighbors
            for i in range(len(site)):
                # Viable neighbor sequence
                move = sequence[:]
                move[site[i]] = AA[i]

                # Make sure this sequence is different.
                if move != sequence:
                    # Store new sequence, fitness and edge.
                    new_moves.append("".join(move))
                    new_fitnesses.append(FF[i])
                    edges.append((("".join(sequence[:]),"".join(move)), {"weight" : prob[i]}))

        # Check that new sequences are actually new.
        moves, indices = np.unique(new_moves, return_index=True)
        fitnesses = np.array(new_fitnesses)[indices]

        # Iterate mutation
        ith_mutation += 1

        # If the number of viable neighbors becomes too large, stop function
        if len(moves) > 5000:
            break

    return edges

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
        print(c)

    p = Process(target=trajectories_true, args=(seq, temp, target, confs, db, directory))
    p.start()
    p.join()

    p1 = Process(target=trajectories_additive, args=(seq, temp, target, confs, db, directory))
    p1.start()
    p1.join()

    p2 = Process(target=trajectories_pairwise, args=(seq, temp, target, confs, db, directory))
    p2.start()
    p2.join()
