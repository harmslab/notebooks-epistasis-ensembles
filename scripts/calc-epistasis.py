__description__ = """\
Script that calculates epistasis across many lattice genotype-phenotype maps in parallel.
"""

__author__ = "Zachary Sailer"


import argparse 
import multiprocessing
import logging
import time

from latticegpm import LatticeGenotypePhenotypeMap
from epistasis.models.transformations import EpistasisLinearTransformation

def worker(i):
    """A single thread for an epistasis calculation on a lattice genotype-phenotype map.
    Calculates epistasis for both stability and native E. Writes the results to json.
    """
    filename = "space-" + str(i) + ".json" 
    path = "results/" + filename
    # Initialize the genotype-phenotype map 
    gpm = LatticeGenotypePhenotypeMap.from_json(path)
    
    # Calculate epistasis on energies
    start = time.time()
    gpm.phenotype_type = "nativeEs"

    model = EpistasisLinearTransformation.from_gpm(gpm, model_type="global")
    model.fit()
    model.epistasis.to_json(path[:-5] + "-Es-coefs.json")
    stop = time.time()
    stopwatch = stop-start
    logging.info("space " + str(i) + " finished nativeEs epistasis. It took " +
                 str(stopwatch) + " seconds.")
    
    # Calculate epistasis on stabilities
    start = time.time()
    gpm.phenotype_type = "stabilities"
    model = EpistasisLinearTransformation.from_gpm(gpm, model_type="global")
    model.fit()
    model.epistasis.to_json("space" + str(i) + "-dG-coefs.json")
    stop = time.time()

    stopwatch = stop-start
    logging.info("space " + str(i) + " finished dG epistasis. It took " + 
                 str(stopwatch) + " seconds.")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="""\
         Read from a folder of json files with\
         genotypes-phenotype maps and calculate epistasis"""
    )
    parser.add_argument("chunk", nargs=2, type=int, help="spaces to iter over.")
    parser.add_argument("--log", default=None, type=str, help="Log file name")
    args = parser.parse_args()
    
    # Set up a log file
    if args.log is None:
        logfile = "job-"+str(args.chunk[0])+"-"+str(args.chunk[1])+".log"
    else:
        logfile = args.log
    logging.basicConfig(filename=logfile,level=logging.DEBUG)
    
    # Check number of cores.
    n_cores = args.chunk[1] - args.chunk[0]
    if n_cores > multiprocessing.cpu_count():
        logging.warning("Trying to use more cores than available.") 
        
    jobs = []
    for i in range(args.chunk[0], args.chunk[1]):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()
        logging.info("Running worker: " + str(i))