import h5py

import numpy as np
import networkx as nx

from epistasis.models.linear import EpistasisLinearRegression
from epistasis.simulate import LinearSimulation, NonlinearSimulation
from epistasis.stats import pearson

from latticeproteins.fitness import Fitness
from latticeproteins.interactions import miyazawa_jernigan
from latticeproteins.sequences import RandomSequence, NMutants
from latticeproteins.conformations import Conformations

from gpmap.utils import hamming_distance
from gpmap.utils import AMINO_ACIDS
from gpmap.graph.paths import path_probabilities, paths_and_probabilities

from latticegpm.gpm import LatticeGenotypePhenotypeMap
from latticegpm.thermo import LatticeThermodynamics
from latticegpm.search import get_lowest_confs

class EvolverError(Exception):
    """"""
    
def fixation(fitness1, fitness2, N=10e8, *args, **kwargs):
    """ Simple fixation probability between two organism with fitnesses 1 and 2.
    Note that N is the effective population size.
    .. math::
        p_{\\text{fixation}} = \\frac{1 - e^{-N \\frac{f_2-f_1}{f1}}}{1 - e^{-\\frac{f_2-f_1}{f1}}}
    """
    sij = (fitness2 - fitness1)/abs(fitness1)
    # Check the value of denominator
    denominator = 1 - np.exp(-N * sij)
    numerator = 1 - np.exp(- sij)
    # Calculate the fixation probability
    fixation = numerator / denominator
    #print(numerator, denominator, fixation)
    return fixation
    
def calculate_phi(model):
    """Calculate the fraction of the phenotypes explained by (statistical) epistasis."""
    # Quick check.
    if model.model_type != "global":
        raise Exception("Argument must be a global model.")
    model.gpm.phenotype_type = "stabilities"
    r2 = [0]
    for i in range(1, model.gpm.binary.length + 1):
        em = model.epistasis.get_orders(*list(range(i+1)))
        vals = np.array(em.values)
        vals[abs(vals) < 1e-13] = 0
        sim = LinearSimulation.from_coefs(
            model.gpm.wildtype,
            model.gpm.mutations,
            em.labels,
            vals,
            model_type=model.model_type)
        sim.sort(model.gpm.genotypes)
        r = pearson(sim.phenotypes, model.gpm.phenotypes)**2
        r2.append(r)
    fx = [r2[i] - r2[i-1] for i in range(2, len(r2))]
    fx = [1 - sum(fx)] + fx
    return fx
    
def adaptive_walk(seq, n_mutations, temp=1.0, target=None):
    """
    """
    length = len(seq)
    c = Conformations(length, "database")
    dGdependence = "fracfolded"

    wildtype = seq
    mutant = list(wildtype)

    indices = list(range(len(wildtype)))    
    path = [wildtype]

    # Get native structure
    fitness = Fitness(temp, c, dGdependence=dGdependence)
    wt_structure = fitness._NativeE(wildtype)[1]

    # Start evolving
    hamming = 0
    while hamming < n_mutations:
        # Calculate stability of all amino acids at all sites
        AA_grid = np.array([AMINO_ACIDS]*length)
        dG = np.zeros(AA_grid.shape, dtype=float)
        for (i,j), AA in np.ndenumerate(AA_grid):
            seq1 = mutant[:]
            seq1[i] = AA_grid[i,j]
            dG[i,j] = fitness.Fitness(seq1)

        # Find the max dG
        x, y = np.where(dG == dG.max())
        best_AA = AA_grid[x[0], y[0]]
        mut = mutant[:]
        mut[x[0]] = best_AA
        
        # Determine native structure mutant
        mut_structure = fitness._NativeE(mut)[1]
        new_hamming = hamming_distance(wildtype, mut)
        
        # Accept move if native structure is maintained and hamming distance increases
        if mut_structure == wt_structure and new_hamming > hamming:
            path.append("".join(mut[:]))
            mutant = mut
            hamming = new_hamming
        else:
            raise EvolverError("No adaptive paths n_mutations away.")
            
    return path

def find_walk(length, n_mutations):
    """Find an adaptive walk in lattice sequence space."""
    while True:
        try:
            wildtype = "".join(RandomSequence(length))
            path = adaptive_walk(wildtype, n_mutations)
            break
        except EvolverError:
            pass
    return path

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
    stuff = []
    for order in range(1, local_model.gpm.binary.length + 1):
        # Build truncated simulation from local model coefs
        em = local_model.epistasis.get_orders(*list(range(order+1)))
        vals = np.array(em.values)
        vals[abs(vals) < 1e-13] = 0
        sim = NonlinearSimulation.from_coefs(
            local_model.gpm.wildtype,
            local_model.gpm.mutations,
            em.labels,
            vals,
            function=frac_folded,
            model_type="local")
        
        sim.sort(local_model.gpm.genotypes)
        Gsim = sim.add_networkx()
        Gsim.add_evolutionary_model(fixation)
        stuff.append(sim.phenotypes)
        
        # Probability of moves away.
        output.append([path_probabilities(Gsim, trajs) for n_mut, trajs in enumerate(subpaths)])
            
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
            if sum(real) == 0 and sum(trunc) == 0:
                r2 = 1
            else:
                r2 = pearson(real, trunc)**2
            if np.isnan(r2):
                r2 = 0
            results.append(r2)
        rsquareds.append(results)
    return rsquareds

if __name__ == "__main__":
    
    f = h5py.File("database-7.hdf5", "w")
    length = 12
    n_mutations = 7
    n_samples = 2

    c = Conformations(length, "database")
    
    sequences = f.create_dataset("sequences", (n_samples,2), dtype="|S" + str(length))
    phi0 = f.create_dataset("phi0", (n_samples, n_mutations), dtype="float64")
    phi1 = f.create_dataset("phi1", (n_samples, n_mutations), dtype="float64")
    phi2 = f.create_dataset("phi2", (n_samples, n_mutations), dtype="float64")
    phi3 = f.create_dataset("phi3", (n_samples, n_mutations), dtype="float64")
    phi4 = f.create_dataset("phi4", (n_samples, n_mutations), dtype="float64")
    rsq0 = f.create_dataset("rsquared0", (n_samples, n_mutations, n_mutations), dtype="float64")
    rsq1 = f.create_dataset("rsquared1", (n_samples, n_mutations, n_mutations), dtype="float64")
    rsq2 = f.create_dataset("rsquared2", (n_samples, n_mutations, n_mutations), dtype="float64")
    rsq3 = f.create_dataset("rsquared3", (n_samples, n_mutations, n_mutations), dtype="float64")
    rsq4 = f.create_dataset("rsquared4", (n_samples, n_mutations, n_mutations), dtype="float64")

    for i in range(n_samples):
                            
        path = find_walk(length, n_mutations)
        wildtype = path[0]
        mutant = path[-1]

        sequences[i] = np.array([wildtype, mutant] , dtype="|S" + str(length))                                   
        cs = get_lowest_confs(wildtype, 3, database="database")
        # Create a genotype-phenotype map for the given sequences (given native target)
        gpm = LatticeGenotypePhenotypeMap.from_mutant(wildtype, mutant, c, temperature=1, target_conf=cs[0])

        ###### BUILD SUBPATHS ONE TIME ########################################################
        if i == 0:
            # Build a list of subpaths that we care about
            G = gpm.add_networkx()
            mapping = gpm.map("genotypes", "indices")
            source = mapping[gpm.wildtype]
            target = mapping[gpm.mutant]

            subpaths = [[] for i in range(gpm.binary.length)]
            for g in gpm.genotypes[1:]:
                target = mapping[g]
                paths = list(nx.all_shortest_paths(G, source, target))
                binary = gpm.binary.genotypes[target]
                n = binary.count("1")-1
                subpaths[n] += (paths)
            

        ###### full-state ensemble! #########################################################
        # Fit with epistasis model
        gpm.phenotype_type = "stabilities"
        lmodel0 = EpistasisLinearRegression.from_gpm(gpm, order=n_mutations, model_type="local")
        lmodel0.fit()

        # Calculate the phi
        gmodel0 = EpistasisLinearRegression.from_gpm(gpm, order=n_mutations, model_type="global")
        gmodel0.fit()
        phi0[i, :] = calculate_phi(gmodel0)

        # Enumerate paths.
        gpm.phenotype_type = "fracfolded"
        summary0 = prob_walk_to_other_genotypes(lmodel0, subpaths)
        rsq0[i, :, :] = rsquared_between_paths(summary0)

        ###### 1-state ensemble! #########################################################
        # Construct subpaths
        gpm.set_partition_confs(cs[0:1])

        # Fit with epistasis model
        gpm.phenotype_type = "nativeEs"
        lmodel1 = EpistasisLinearRegression.from_gpm(gpm, order=n_mutations, model_type="local")
        lmodel1.fit()

        # Calculate the phi
        gmodel1 = EpistasisLinearRegression.from_gpm(gpm, order=n_mutations, model_type="global")
        gmodel1.fit()
        phi1[i,:] = calculate_phi(gmodel1)

        # Enumerate paths.
        gpm.phenotype_type = "fracfolded"
        summary1 = prob_walk_to_other_genotypes(lmodel1, subpaths)
        rsq1[i, :, :] = rsquared_between_paths(summary1)

        ###### 2-state ensemble! #########################################################
        # Construct subpaths
        gpm.set_partition_confs(cs[0:2])

        # Fit with epistasis model
        gpm.phenotype_type = "stabilities"
        lmodel2 = EpistasisLinearRegression.from_gpm(gpm, order=n_mutations, model_type="local")
        lmodel2.fit()

        # Calculate the phi
        gmodel2 = EpistasisLinearRegression.from_gpm(gpm, order=n_mutations, model_type="global")
        gmodel2.fit()
        phi2[i,:] = calculate_phi(gmodel2)

        # Enumerate paths.
        gpm.phenotype_type = "fracfolded"
        summary2 = prob_walk_to_other_genotypes(lmodel2, subpaths)
        rsq2[i, :, :] = rsquared_between_paths(summary2)

        ###### 3-state ensemble! #########################################################
        # Construct subpaths
        gpm.set_partition_confs(cs)

        # Fit with epistasis model
        gpm.phenotype_type = "stabilities"
        lmodel3 = EpistasisLinearRegression.from_gpm(gpm, order=n_mutations, model_type="local")
        lmodel3.fit()

        # Calculate the phi
        gmodel3 = EpistasisLinearRegression.from_gpm(gpm, order=n_mutations, model_type="global")
        gmodel3.fit()
        phi3[i,:] = calculate_phi(gmodel3)

        # Enumerate paths.
        gpm.phenotype_type = "fracfolded"
        summary3 = prob_walk_to_other_genotypes(lmodel3, subpaths)
        rsq3[i, :, :] = rsquared_between_paths(summary3)

        ###### 4-state ensemble! #########################################################
        # Construct subpaths
        gpm.set_partition_confs(cs)

        # Fit with epistasis model
        gpm.phenotype_type = "stabilities"
        lmodel4 = EpistasisLinearRegression.from_gpm(gpm, order=n_mutations, model_type="local")
        lmodel4.fit()

        # Calculate the phi
        gmodel4 = EpistasisLinearRegression.from_gpm(gpm, order=n_mutations, model_type="global")
        gmodel4.fit()
        phi4[i,:] = calculate_phi(gmodel4)

        # Enumerate paths.
        gpm.phenotype_type = "fracfolded"
        summary4 = prob_walk_to_other_genotypes(lmodel4, subpaths)
        rsq4[i, :, :] = rsquared_between_paths(summary4)
    