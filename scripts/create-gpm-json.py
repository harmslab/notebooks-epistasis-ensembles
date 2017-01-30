import argparse 
import multiprocessing
import logging

from latticeproteins import Conformations
from latticegpm import LatticeGenotypePhenotypeMap

def read_list_of_seq_pairs(filename):
    """Reads a list of sequence pairs from file, 
    and returns them as a python list of tuples.
    """
    with open(filename, "r") as f:
        # sorry for the cryptic one liner
        sequences = [tuple(line.strip().split(",")) for line in f.lines()]
    return sequences

def write_gpm(seq1, seq2, filename):
    """Must have a conformations database in a subdirectory.
    """
    conformations = Conformations("database/")
    gpm = LatticeGenotypePhenotypeMap.from_mutant(seq1, seq2)
    gpm.to_json(filename)
    return gpm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Must have a database directory.")
    parser.add_argument("filename", help="file with a list of sequence pairs.")
    parser.add_argument("chunk", type=int, help="sequences to iter over.")
    parser.add_argument("-o", default=None, type=str, help="output file name")
    parser.add_argument("--log", default=None, type=str, help="Log file name")
    args = parser.parse_args()

    # Set up a log file
    if args.log is None:
        logfile = "job-"+args.chunk[0]+"-"+args.chunk[1]+".log"
    else:
        logfile = "job.log"
    logging.basicConfig(filename=logfile,level=logging.DEBUG)
    
    # read sequences and use subset
    sequences = read_list_of_seq_pairs(args.filename)
    seqs_to_use = sequences[args.chunk[0], args.chunk[1]]
    
    # Check number of cores.
    n_cores = len(seqs_to_use)
    if n_cores > multiprocessing.cpu_count():
        logging.warning("Trying to use more cores than available.")
    
    jobs = []
    for i in range(n_cores):
        seq1 = seqs_to_use[i][0]
        seq2 = seqs_to_use[i][1]
        line_n = args.chunk[0] + i
        output_name = "space-" + str(line_n) + ".json"
        p = multiprocessing.Process(target=write_gpm, args=(seq1, seq2, output_name))
        jobs.append(p)
        p.start()
        logging.info("running")
    