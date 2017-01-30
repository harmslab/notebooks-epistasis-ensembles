__description__ = """\
Samples forward trajectories through the binary genotype-phenotype map between
two lattice protein sequences. Two sets of trajectories are calculated, one with
its full conformational ensemble, and the other with only a two state conformational
ensemble. We compare these two sets to observe the role of epistasis that arises
from thermodynamic ensembles.

Writes the results as a pickled Counter object. The keys are tuples of paths seen,
and the values are the number of counts for that path.
"""

__author__ = "Zachary Sailer"

import os
import argparse
import multiprocessing
import logging
import time
import pickle

from collections import Counter

from latticegpm import LatticeGenotypePhenotypeMap
from latticegpm.evolve.models import fixation
from latticegpm.evolve import monte_carlo

def get_gpm(wildtype, mutant):
    """Get a genotype-phenotype map from a wildtype sequence and mutant sequence.
    """
    gpm1 = LatticeGenotypePhenotypeMap.from_mutant(wildtype, mutant)
    gpm1.phenotype_type = "fracfolded"

    # Calculate same genotype without an ensemble.
    gpm2 = LatticeGenotypePhenotypeMap.from_mutant(wildtype, mutant)
    gpm2.phenotype_type = "fracfolded"

    # recalculate partition function
    return gpm1, gpm2

def add_samples_to_counter(gpm, counter, n_samples=100):
    """Takes a trajectory counter, sample n_samples trajectories using Monte Carlo
    methods, and add n_samples to the counter.
    """
    # Identify wildtype and mutant
    mapping = gpm.map("genotypes", "indices")
    source = mapping[gpm.wildtype]
    target = mapping[gpm.mutant]

    paths = []
    for i in range(n_samples):
        # Calculate a path
        paths.append( monte_carlo(gpm, source, target, fixation, forward=True) )

    # Add new paths to counter
    c.update(path)
    return c

def worker(gpm, counter_file, n_samples=10000, chunk_size=100):
    """Creates counter files (or updates them if they already exist), samples
    evolutionary trajectories through a genotype-phenotype map.

    Parameters
    ----------
    gpm : LatticeGenotypePhenotypeMap
        lattice protein genotype-phenotype map.
    counter_prefix : str
        prefix name for counter files.
    n_samples : int (default=10000)
        Number of total samples to add
    chunk_size : int (default=100)
        Chunk size, break-points for writing to counter files during long runs.
    """
    ######## SANITY CHECKS ########
    # Check if a running counter exists.
    try:
        with open(counter_file, "rb") as f:
            c = pickle.load(f)
    except FileNotFoundError:
        c = Counter()

    ######## MAIN CODE ########
    # Run one set of samples and time it.
    start = time.time()
    c = add_samples_to_counter(gpm, c, n_samples)
    stop = time.time()

    logging.info("Time to complete %d sized chunk for %s : %.3f" % (chunk_size, counter_file, stop-start))

    # Run all samples and time that.
    for i in range(chunk_size, n_samples, chunk_size):
        c = add_samples_to_counter(gpm, c, chunk_size)
        # Write chunk to file.
        with open(counter_file, "wb") as f:
            pickle.dump(f, c)

    stop = time.time()
    logging.info("Time to complete %d samples for no ensemble for %s : %.3f" % (n_samples, counter_file, stop-start))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("prefix", type=str, help="counter file prefix")
    parser.add_argument("n_samples", type=int, help="Number of samples.")
    parser.add_argument("chunk_size", default=100, type=int, help="spaces to iter over.")
    parser.add_argument("--log", default=None, type=str, help="Log file name")

    # Parse arguments.
    args = parser.parse_args()

    # Set up a log file
    if args.log is None:
        logfile = "logs/job-"+str(args.chunk[0])+"-"+str(args.chunk[1])+".log"
    else:
        logfile = args.log
    logging.basicConfig(filename=logfile, level=logging.DEBUG)

    # Check number of cores.
    n_cores = (args.chunk[1] - args.chunk[0]) * 2
    if n_cores > multiprocessing.cpu_count():
        logging.warning("Trying to use more cores than available.")

    # Read sequences file.
    with open("sequences.txt", "r") as f:
        sequences = []
        for line in f.readlines():
            seqs = tuple(line.split(","))
            seqs[0] = seq[0].strip()
            seqs[1] = seq[1].strip()
            sequences.append(seqs)

    # File name.
    f1 = "results/" + args.prefix + "-z.pickle"
    f2 = "results/" + args.prefix + "-no-z.pickle"

    jobs = []
    for i in range(args.chunk[0], args.chunk[1]):
        # Get wildtype and mutant
        seq0, seq1 = sequences[i][0], sequences[i][1]

        start = time.time()
        # Read a set of genotypes
        gpm1, gpm2 = get_gpm(seq0, seq1)
        stop = time.time()
        logging.info("Time to build two genotype-phenotype maps : %03f" % (stop - start,))

        # Start processes for this space.
        p = multiprocessing.Process(target=worker, args=(gpm1, f1,))
        p2 = multiprocessing.Process(target=worker, args=(gpm2, f2,))

        # Start jobs.
        jobs.append(p)
        jobs.append(p2)
        p.start()
        p2.start()
