import numpy as np
import networkx as nx
import pickle
import argparse

from gpmap.utils import binary_mutations_map, find_differences
from latticegpm import LatticeGenotypePhenotypeMap
from latticeproteins.conformations import Conformations, ConformationList
from epistasis.models import EpistasisLinearRegression
from epistasis.simulate import NonlinearSimulation
from gpvolve.evolve.models import fixation
from gpgraph import GenotypePhenotypeGraph
from gpgraph.draw import network
from gpgraph.paths import paths_and_probabilities

def overlap(p,q):
    """Calculates the difference in overlap."""
    return np.array(abs(p - q)).sum() / (p.sum() + q.sum())

def epistasis_between_pair(pair, target, db=None):
    """Calculate the epistasis between two sequences

    Returns
    -------
    epistasis : list
        magnitude of epistasis for each order. First element is the
        additive terms, second is pairwise epistasis, so on...
    order : dict
        Keys are order of model used. Values are list of differences (theta) between
        the known path probabilities and the simulated (using epistasis model)
        at each mutational step away.
    """
    popsize = 1000

    # Prepare for GenotypePhenotypeMap
    wildtype = pair[0]
    length = len(wildtype)
    mutations = binary_mutations_map(*pair)

    # Determine the order of epistasis
    changes = find_differences(*pair)
    order = len(changes)

    if db is None:
        c = Conformations(length)
    else:
        c = ConformationList(length, db)

    # Construct a genotype-phenotype map
    gpm = LatticeGenotypePhenotypeMap(wildtype, mutations=mutations, target_conf=target)
    gpm.fold(Conformations=c)

    # Fit a high-order epistasis model
    model = EpistasisLinearRegression.read_gpm(gpm, order=order, model_type="local")
    model.fit()

    # Calculate the average magnitude of epistasis
    epistasis = []
    for i in range(1,order+1):
        z = model.epistasis.get_orders(i)
        coefs_ = np.abs(z.values)
        # Protect numerical errors
        coefs_[coefs_ < 1e-9] = 0
        # Calculate mean
        mcoefs_ = np.mean(coefs_)
        epistasis.append(mcoefs_)

    # Model evolution in the map
    gpm.phenotype_type = "fracfolded"
    G = GenotypePhenotypeGraph(gpm)
    G.add_evolutionary_model(fixation, N=popsize)

    paths1 = enumerate_all_subpaths(G)

    # Build a predicted genotype-phenotype map from epistasis
    overlaps = {}
    for i in range(1, 8):
        # Get subset of epistasis
        em = model.epistasis.get_orders(*range(i+1))

        # Build map from epistatic coefs
        sim = NonlinearSimulation.from_coefs(
            wildtype, mutations, em.sites, em.values,
            function=lambda x: 1 / (1 + np.exp(x)),
            model_type="local")

        # Build a network
        Gsim = GenotypePhenotypeGraph(sim)

        # Model evolution in simulation
        Gsim.add_evolutionary_model(fixation, N=popsize)
        paths2 = enumerate_all_subpaths(Gsim)

        overlaps[i] = overlap_for_subpaths(paths1, paths2)

    return epistasis, overlaps

def enumerate_all_subpaths(G):
    """Enumerate all paths to all nodes in the network, G. Then, calculate their
    flux.

    Parameter
    ---------
    G : networkx graph
        Graph with wildtype labeled as 0, and each edge has a `fixation` key.

    Returns
    -------
    paths : dict of dicts
        Keys are number of mutational steps, values are dictionaries that have the path
        mapped to the path's probability (product of fixation probabilties).

    Example
    -------
    paths = {
        1 : {(0,3): 0.6, (0,2): 0.4 ... }
        2 : {(0,3,6) : 0.51, (0,2,7), ... }
    }
    """
    source = 0
    nodes = list(G.nodes())
    nodes.remove(source)

    # Begin enumerate paths to all nodes
    paths = dict([(i, {}) for i in range(1, 7)])
    for target in nodes:
        # enumerate all paths to a given target
        pathways = list(nx.all_shortest_paths(G, source, target))
        nmuts = len(pathways[0])-1

        # Get flux from all paths
        probabilities = {}
        for path in pathways:
            path = tuple(path)

            # Don't repeat any calcs, use old calc and
            if nmuts > 1:
                current_prob = paths[nmuts-1][path[:nmuts]]
                new_move_prob = G.edge[path[-2]][path[-1]]["fixation"]

                # Calculate the probability of this new step
                pi = current_prob * new_move_prob
                probabilities[path] = pi
            else:
                probabilities[path] = G.edge[path[-2]][path[-1]]["fixation"]

        # Add these paths/probs to paths dictionary.
        paths[nmuts].update(probabilities)

    return paths

def overlap_for_subpaths(paths1, paths2):
    """"""
    overlaps = []
    for nsteps in range(1, len(paths1)+1):
        pathways1 = paths1[nsteps]
        pathways2 = paths2[nsteps]
        # Built a spectrum of probabilities
        probs1 = np.array([pathways1[key] for key in pathways1.keys()])
        probs2 = np.array([pathways2[key] for key in pathways2.keys()])
        # Calculate overlap
        theta = overlap(probs1, probs2)
        overlaps.append(theta)
    return overlaps


def main(start, stop, n_states):
    # Directory to pull/push results
    if n_states == 0:
        dir = "full-state"
    
        # Read in actual dataset
        with open("results-{}/binary-input.pickle".format(dir), "rb") as f:
            data = pickle.load(f)
            gpms = data["seq_pairs"]
            targets = data["targets"]
            dbs = data["dbs"][0:n_states]
    
    else:
        dir = "{}-state".format(n_states)
        # Read in actual dataset
        with open("results-{}/binary-input.pickle".format(dir), "rb") as f:
            data = pickle.load(f)
            gpms = data["seq_pairs"]
            targets = data["targets"]
            dbs = None
    
    size = stop - start
    magn_epistasis = np.ones((size, 6), dtype=float) * np.nan
    overlaps = dict([(i, []) for i in range(1, 8)])

    # Read in actual dataset
    with open("results-{}/binary-input.pickle".format(dir), "rb") as f:
        data = pickle.load(f)
        gpms = data["seq_pairs"]
        targets = data["targets"]
        dbs = data["dbs"]
        
        
    for key, pair in gpms.items():
        epist, over = epistasis_between_pair(pair, targets[key], dbs[key][0:2])
        magn_epistasis[key, :len(epist)] = epist

        for order in over:
            overlaps[order].append(over[order])

    out = {
        "epistasis" : magn_epistasis,
        "thetas" : overlaps
    }
    with open("results-{}/binary-output.pickle".format(dir), "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    # Handle command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs=2, help="datasets to compare")
    parser.add_argument("n_states", type=int, help="Number of states in ensemble, 0 means full.")
    args = parser.parse_args()
    start = args.n[0]
    stop = args.n[1]

    main(start, stop, args.n_states)
