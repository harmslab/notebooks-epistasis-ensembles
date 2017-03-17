__doc__ = """
Search lattice model sequence space.

This script is difficult to follow... sorry for the lack of simplicity. may
return to this later. Runs as is.

Output is clearly explained below.

Output
------

"""

# ---------------------------------
# Imports
# ---------------------------------

import numpy as np
import h5py
import networkx as nx

from epistasis.models.linear import EpistasisLinearRegression
from epistasis.simulate import LinearSimulation, NonlinearSimulation
from epistasis.stats import pearson

from latticeproteins.sequences import RandomSequence

from latticegpm import LatticeGenotypePhenotypeMap
from latticegpm.thermo import LatticeThermodynamics
from latticegpm.search import adaptive_walk, get_lowest_confs

from gpmap.evolve.models import fixation
from gpmap.graph.paths import path_probabilities, paths_and_probabilities

def calculate_phi(model):
    """Calculate the fraction of the phenotypes explained by (statistical) epistasis."""
    # Quick check.
    if model.model_type != "global":
        raise Exception("Argument must be a global model.")
    model.gpm.phenotype_type = "stabilities"
    r2 = [0]
    for i in range(1,9):
        em = model.epistasis.get_orders(*list(range(i+1)))
        sim = LinearSimulation.from_coefs(
            model.gpm.wildtype,
            model.gpm.mutations,
            em.labels,
            em.values,
            model_type=model.model_type)
        r = pearson(sim.phenotypes, model.gpm.phenotypes)**2
        r2.append(r)
    fx = [r2[i] - r2[i-1] for i in range(2, len(r2))]
    fx = [1 - sum(fx)] + fx
    return fx

def prob_walk_to_other_genotypes(local_model, subpaths):
    """Calculate the probability of walking to nodes away from ancestor.

    - Output is a list of lists of lists containing (model-order, probabilites).
    - output[0][0] is a list predicted probabilities using model 0 to genotypes
      1 mutation away.
    """
    # Quick check.
    if local_model.model_type != "local":
        raise Exception("Argument must be a local model.")

    def frac_folded(x):
        return 1 / (1 + np.exp(x))

    output = []
    for order in range(1, 9):
        # Build truncated simulation from local model coefs
        em = local_model.epistasis.get_orders(*list(range(order+1)))
        sim = NonlinearSimulation.from_coefs(
            local_model.gpm.wildtype,
            local_model.gpm.mutations,
            em.labels,
            em.values,
            function=frac_folded,
            model_type="local")
        mapping = sim.map("genotypes", "indices")
        source = mapping[wildtype]
        Gsim = sim.add_networkx()
        Gsim.add_evolutionary_model(fixation)

        # Probability of moves away.
        probs = []
        for n_mut, trajs in enumerate(subpaths):
            probs.append(path_probabilities(Gsim, trajs))
        output.append(probs)

    return output

def rsquared_between_paths(summary):
    """Compare truncated model trajectory predictions to the full order model.

    Summary is the output from `prob_walk_to_other_genotypes`
    """
    rsquareds = []
    actual = summary[-1]
    for i, trajs in enumerate(summary):
        results = []
        for n_muts, trunc_ in enumerate(summary[i]):
            real = np.array(actual[n_muts])
            trunc = np.array(trunc_)
            r2 = pearson(real, trunc)**2
            if np.isnan(r2):
                r2 = 0
            results.append(r2)
        rsquareds.append(results)
    return rsquareds


if __name__ == "__main__":
    length = 12
    n_muts = 8
    n_samples = 100

    # prepare file
    f = h5py.File("lattice-8site.hdf5", "w")
    sequences = f.create_dataset("sequences", (n_samples, 2), dtype="|S" + str(length))
    phi2 = f.create_dataset("phi-2", (n_samples, n_muts), dtype="float64")
    rsquared2 = f.create_dataset("rsquared-2", (n_samples, n_muts, n_muts), dtype="float64")
    phi3 = f.create_dataset("phi-3", (n_samples, n_muts), dtype="float64")
    rsquared3 = f.create_dataset("rsquared-3", (n_samples, n_muts, n_muts), dtype="float64")

    current = 0
    n_failures = 0

    while current < n_samples:

        while True:
            wildtype = "".join(RandomSequence(length))
            cs = get_lowest_confs(wildtype, 3, database="database")
            wt = LatticeThermodynamics(wildtype, conf_list=cs[0:2], temperature=1)
            try:
                mut = adaptive_walk(wt, n_muts)
                mutant = mut.sequence
                break
            except: continue
        # Save sequences
        sequences[current] = np.array([wildtype, mutant] , dtype="|S" + str(length))

        # Find the lost conformations in the free energy landscape of the given sequences.
        wt = LatticeThermodynamics(wildtype, conf_list=cs[0:2], temperature=1)
        mut = LatticeThermodynamics(mutant, conf_list=cs[0:2], temperature=1)

        # Create a genotype-phenotype map for the given sequences (given native target)
        gpm = LatticeGenotypePhenotypeMap.from_Lattice(wt, mut, temperature=1, target_conf=cs[0])
        gpm.phenotype_type = "stabilities"

        # Build a list of subpaths that we care about
        G = gpm.add_networkx()
        mapping = gpm.map("genotypes", "indices")
        source = mapping[wildtype]
        target = mapping[mutant]

        if True:
            ###### 2-state ensemble!
            # Construct subpaths
            if current == 0:
                subpaths = [[] for i in range(8)]
                for g in gpm.genotypes[1:]:
                    target = mapping[g]
                    paths = list(nx.all_shortest_paths(G, source, target))
                    binary = gpm.binary.genotypes[target]
                    n = binary.count("1")-1
                    subpaths[n] += (paths)

            # Fit with epistasis model
            gpm.phenotype_type = "stabilities"
            lmodel2 = EpistasisLinearRegression.from_gpm(gpm, order=8, model_type="local")
            lmodel2.fit()

            # Calculate the phi
            gmodel2 = EpistasisLinearRegression.from_gpm(gpm, order=8, model_type="global")
            gmodel2.fit()
            phi2[current,:] = calculate_phi(gmodel2)

            # Enumerate paths.
            summary2 = prob_walk_to_other_genotypes(lmodel2, subpaths)
            rsquared2[current,:,:] = rsquared_between_paths(summary2)

            ###### 3-state ensemble!
            gpm.set_partition_confs(cs)

            # Fit with epistasis model
            gpm.phenotype_type = "stabilities"
            lmodel3 = EpistasisLinearRegression.from_gpm(gpm, order=8, model_type="local")
            lmodel3.fit()

            # Calculate the phi
            gmodel3 = EpistasisLinearRegression.from_gpm(gpm, order=8, model_type="global")
            gmodel3.fit()
            phi3[current,:] = calculate_phi(gmodel3)

            # Enumerate paths.
            summary3 = prob_walk_to_other_genotypes(lmodel3, subpaths)
            rsquared3[current,:,:] = rsquared_between_paths(summary3)

            current += 1
            print(current)
            with open("sequences.txt", "a") as f:
                f.write(wildtype + ", " + mutant +"\n")
        else:
            n_failures += 1
           
