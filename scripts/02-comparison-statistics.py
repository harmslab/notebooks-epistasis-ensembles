import numpy as np
import networkx as nx
import pickle
import argparse

from latticeproteins.sequences import hamming_distance


def ring_levels(G, root):
    levels = dict([(i,[]) for i in range(20)])
    levels[0].append(root)
    for node in G.nodes():
        neighbors = G.neighbors(node)
        for neigh in neighbors:
            key = hamming_distance(root, neigh)
            levels[key].append(neigh)
    for key, val in levels.items():
        z = sorted(list(set(val)))
        levels[key] = z
    return levels


def top_path(G, source):
    # Find moves away from center
    rings = ring_levels(G, source)
    # Get further derived sequences
    for i in range(7):
        if len(rings[i]) != 0:
            furthest = i
    ends = rings[furthest] # only 6 moves away

    # Build a dictionary of node flux to
    options = dict([(end, G.node[end]['flux']) for end in ends])

    sequences = np.array(list(options.keys()))
    fluxes = np.array(list(options.values()))

    # Find the highest flux sequence
    most_probable_node = np.argmax(fluxes)

    max_seq = sequences[most_probable_node]

    # Return that sequence
    return (source, max_seq)

def main(start, stop, dir):

    bad_datasets = []

    size = stop - start

    dfluxes = dict([(i, np.ones(size)*np.nan) for i in range(8)])
    gpms = {}
    targets = {}
    dbs = {}

    for d in range(start, stop):
        # Formulate as networks
        try:
            with open("results-{}/networks-{}.pickle".format(dir, d), "rb") as f:
                data = pickle.load(f)
                G0 = data["Gactual"]
                G2 = data["Gpredict"]
                Gdiff = data["Gdiff"]

            with open("results-{}/trajectories-true-{}.pickle".format(dir, d), "rb") as f:
                data = pickle.load(f)
                seq = data["seq"]
                target = data["target"]
                temp = data["temp"]
                db = data["db"]

            rings = ring_levels(Gdiff, seq)

            # Add to summary statistic
            for ring, sequences in rings.items():

                if  ring < 8:

                    diff, denom1, denom2 = 0, 0, 0
                    for i in sequences:

                        # Add up total flux exchange, and the difference between two networks
                        for j in Gdiff.neighbors(i):

                            diff += Gdiff.edge[i][j]["weight"]
                            try: denom1 += G0.edge[i][j]["delta_flux"]
                            except: pass

                            try: denom2 += G2.edge[i][j]["delta_flux"]
                            except: pass

                    denom = denom1 + denom2
                    #print(d, ring, denom, diff, seq)
                    # Normalize the flux exchange at this hamming distance
                    if denom != 0:
                        dfluxes[ring][d] = diff/denom

            gpms[d] = top_path(G0, seq)
            targets[d] = target
            dbs[d] = db
        except Exception as e:
            bad_datasets.append((d, e))


    with open("results-{}/statistics-output.pickle".format(dir), "wb") as f:
        pickle.dump(dfluxes,f)

    out = {
        "seq_pairs" : gpms,
        "targets" : targets,
        "dbs" : dbs
    }
    with open("results-{}/binary-input.pickle".format(dir), "wb") as f:
        pickle.dump(out,f)

    if len(bad_datasets) > 0:
        # Write bad dataset numbers to file.
        with open("bad_datasets.txt", "a") as f:
            for item in bad_datasets:
                f.write("%d\t%s\n" % (item[0], item[1]))


if __name__ == "__main__":
    # Handle command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs=2, help="datasets to compare")
    parser.add_argument("dir", type=str, help="directory to run script")
    args = parser.parse_args()
    start = args.n[0]
    stop = args.n[1]

    main(start, stop, args.dir)
