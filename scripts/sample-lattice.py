__doc__ = """\
Create a database of 2d lattice proteins with two different \
thermodynamic ensembles: 2-state and 3-state partition functions. \

The purpose of this database is to study high-order epistasis \
that arises from the effect of mutation on many states in the \
ensemble. \n

Lattice sequences are selected using the following criterion: \
1. given length and number of mutations between them. \
2. same native state \
3. 3-state ensemble exhibits high-order epistasis (2-state should never exhibit HOE) \
4. trajectories are accessible in both spaces. \
"""

import argparse
import h5py

import numpy as np

from latticeproteins.conformations import Conformations
from latticeproteins.sequences import RandomSequence

from gpmap.evolve.models import fixation
from gpmap.graph.paths import paths_and_probabilities
from gpmap.graph.paths import flux
from gpmap.graph import draw

import latticegpm
from latticegpm import LatticeGenotypePhenotypeMap
from latticegpm.thermo import LatticeThermodynamics
from latticegpm.search import adaptive_walk

from epistasis.models import EpistasisLinearRegression

def select_sequences(length, n_mutations):
    """Get wildtype and mutant sequence -- with their native state and
    2-state and 3-state ensembles

    length : length of sequences
    n_mutations : number of mutations between sequences
    """
    wildtype = "".join(RandomSequence(length))
    c = Conformations(length, "database")
    dGdependence = "fracfolded"
    temperature = 1
    # Calculate the kth lowest conformations
    ncontacts = c.MaxContacts()
    confs = np.array(c.UniqueConformations(ncontacts))
    energies = np.empty(len(confs), dtype=float)
    for i, conf in enumerate(confs):
        output = c.FoldSequence(wildtype, temperature, target_conf=str(conf), loop_in_C=False)
        energies[i] = output[0]

    # Get states
    sorted_e = np.argsort(energies)
    two_states = confs[sorted_e[0:2]]
    three_states = confs[sorted_e[0:3]]

    # Find mutant
    wtlattice = LatticeThermodynamics(wildtype, three_states, temperature=temperature)
    mutlattice = adaptive_walk(wtlattice, n_mutations)
    native = mutlattice.native_conf
    mutant = mutlattice.sequence

    return wildtype, mutant, native, two_states, three_states

def build_gpm(wildtype, mutant, native, states):
    """Build genotype-phenotype map.
    """
    # Construct a genotype-phenotype map
    length = len(wildtype)
    c = Conformations(length, "database")
    gpm = LatticeGenotypePhenotypeMap.from_mutant(wildtype, mutant, c,
        temperature=1,
        target_conf=native)
    gpm.phenotype_type = "fracfolded"
    gpm.set_partition_confs(states)

    # Build graph
    G = gpm.add_networkx()
    G.add_evolutionary_model(fixation)

    return gpm, G

def calculate_epistasis(gpm):
    """Estimate epistasis and the fraction explained by each
    order of epistatic coefficients.
    """
    gpm.phenotype_type = "stabilities"
    model = EpistasisLinearRegression.from_gpm(gpm, order=0, model_type="global")
    model.fit()
    fx = [model.score()]
    for i in range(1, gpm.binary.length+1):
        model = EpistasisLinearRegression.from_gpm(gpm, order=i, model_type="global")
        model.fit()
        score = model.score() - sum(fx)
        fx.append(score)
    return model, fx

def sample(i, f, length, n_mutations):
    failed = False
    # Select sequences
    wildtype, mutant, native, two_states, three_states = select_sequences(length, n_mutations)

    # Get genotype-phenotype maps with 2 state and 3 state ensembles
    gpm_2state, G_2state = build_gpm(wildtype, mutant, native, two_states)
    gpm_3state, G_3state = build_gpm(wildtype, mutant, native, three_states)

    # Find source and target nodes
    mapping = gpm_3state.map("genotypes", "indices")
    s = mapping[wildtype]
    t = mapping[mutant]

    # Calculate paths
    paths_2state, probs_2state = paths_and_probabilities(G_2state, s, t)
    paths_3state, probs_3state = paths_and_probabilities(G_3state, s, t)

    # Calculate epistasis in these spaces
    model2, fx2 = calculate_epistasis(gpm_2state)
    model3, fx3 = calculate_epistasis(gpm_3state)

    # Checks
    check1 = sum(fx3[3:]) == 0
    check2 = sum(probs_2state) == 0
    check3 = sum(probs_3state) == 0

    if True not in (check1, check2, check3):
        # Write to file
        f["wildtype"][i] = wildtype.encode("utf8")
        f["mutant"][i] = mutant.encode("utf8")
        f["states_2"][:,i] = [s.encode("utf8") for s in two_states]
        f["states_3"][:,i] = [s.encode("utf8") for s in three_states]
        f["traj_2"][:,i] = probs_2state
        f["traj_3"][:,i] = probs_3state
        f["epistasis_2"][:,i] = model2.epistasis.values
        f["epistasis_3"][:,i] = model3.epistasis.values
        f["fx_2"][:,i] = fx2
        f["fx_3"][:,i] = fx3
    else:
        failed = True
    return failed

def main(filename, length, n_mutations, n_samples):
    f = h5py.File(filename, "w")
    f.create_dataset("wildtype", (n_samples,), dtype="|S"+str(length))
    f.create_dataset("mutant", (n_samples,), dtype="|S"+str(length))
    f.create_dataset("states_2", (2, n_samples), dtype="|S"+str(length-1))
    f.create_dataset("states_3", (3, n_samples), dtype="|S"+str(length-1))
    f.create_dataset("traj_2", (120, n_samples), dtype="float64")
    f.create_dataset("traj_3", (120, n_samples), dtype="float64")
    f.create_dataset("epistasis_2", (32, n_samples), dtype="float64")
    f.create_dataset("epistasis_3", (32, n_samples), dtype="float64")
    f.create_dataset("fx_2", (6, n_samples), dtype="float64")
    f.create_dataset("fx_3", (6, n_samples), dtype="float64")

    i = 0
    while i < n_samples:
        try:
            failed = sample(i, f, length, n_mutations)
            if failed is False:
                i += 1
        except ValueError:
            pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", type=str,
        help="Filename of hdf5 output file.")
    parser.add_argument("length", type=int,
        help="Length of sequences.")
    parser.add_argument("n_mutations", type=int,
        help="Number of mutations between ancestor and derived sequences.")
    parser.add_argument("n_samples", type=int,
        help="Number of samples")
    args = parser.parse_args()

    main(args.filename, args.length, args.n_mutations, args.n_samples)
