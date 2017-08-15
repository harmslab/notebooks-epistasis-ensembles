import numpy as np
import networkx as nx
import pickle
import argparse
from latticeproteins.sequences import hamming_distance

# ----------------------------------------------------------
# Main Code
# ----------------------------------------------------------

def main(start, stop, dir):
    """
    """
    # List to collect datasets we are unable to analyze.
    bad_datasets = []

    # Iterate through many dataset
    for d in range(start, stop):
        # Read in actual dataset
        with open("results-{}/trajectories-true-{}.pickle".format(dir, d), "rb") as f:
            data = pickle.load(f)
            edges0 = data["edges"]
            seq = data["seq"]
            target = data["target"]
            temp = data["temp"]
            db = data["db"]

        # Read in predicted dataset
        with open("results-{}/trajectories-pairwise-{}.pickle".format(dir, d), "rb") as f:
            data = pickle.load(f)
            edges2 = data["edges"]

        # Construct as networks
        try:
            Gactual, Gpredict, Gdiff = build_graphs(edges0, edges2, seq)
            data = {
                "Gactual" : Gactual,
                "Gpredict" : Gpredict,
                "Gdiff" : Gdiff,
                "seq" : seq
            }
            with open("results-{}/networks-{}.pickle".format(dir, d), "wb") as f:
                pickle.dump(data, f)
        # Collect bad datasets for debugging later
        except Exception as e:
            bad_datasets.append((d,e))

    if len(bad_datasets) > 0:
        # Write bad dataset numbers to file.
        with open("bad_datasets.txt", "a") as f:
            for item in bad_datasets:
                f.write("%d\t%s\n" % (item[0], item[1]))

# ----------------------------------------------------------
# Functions used in this code.
# ----------------------------------------------------------

def flux_out_of_node(G, node_i):
    """Determine """
    # Get flux coming from source
    total_flux_avail = G.node[node_i]["flux"]
    edges = {}
    # Normalize the transition probability from source
    norm = sum([G.edge[node_i][node_j]["weight"] for node_j in G.neighbors(node_i)])
    # Iterate over neighbors divvy up flux across neighbors
    for node_j in G.neighbors(node_i):
        if norm > 0:
            fixation = G.edge[node_i][node_j]["weight"]
            dflux = (fixation/norm) * total_flux_avail
            G.edge[node_i][node_j]["delta_flux"] = dflux
            G.node[node_j]["flux"] += dflux
        else:
            G.edge[node_i][node_j]["delta_flux"] = 0
    return edges

def flux_from_source(G, source):
    # Reset the flux of each node
    init_flux = dict([(node, 0) for node in G.nodes()])
    nx.set_node_attributes(G, "flux", init_flux)
    G.node[source]["flux"] = 1
    # Add flux to each node.
    levels = ring_levels(G, source)
    for l in levels:
        for node_i in levels[l]:
            edges = flux_out_of_node(G, node_i)
            for key, flux_to_add in edges.items():
                node_i, node_j = key
                G.node[node_j]["flux"] += flux_to_add
    return G


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


def build_graphs(edges1, edges2, source):
    """Construct two different networks from a set of edges.
    """
    # -----------------------------------------------
    # build initial graphs
    # -----------------------------------------------
    edges0 = edges1
    # Build Graph
    G0 = nx.DiGraph()
    for key, weight in edges0:
        i,j = key[0], key[1]
        G0.add_edge(i,j, weight=weight["weight"])

    # Build Graph
    G2 = nx.DiGraph()
    for key, weight in edges2:
        i,j = key[0], key[1]
        G2.add_edge(i,j, weight=weight["weight"])

    seq = source
    # -----------------------------------------------
    # Calculate the flux at each node and edge
    # -----------------------------------------------
    G0 = flux_from_source(G0, seq)
    G2 = flux_from_source(G2, seq)

    # Get a dictionary of change in fluxes along each edge.
    edges_0 = dict([((i, j), G0.edge[i][j]["delta_flux"]) for i,j in G0.edges()])
    edges_2 = dict([((i, j), G2.edge[i][j]["delta_flux"]) for i,j in G2.edges()])

    # -----------------------------------------------
    # Calculate the change in delta_flux on each edge
    # -----------------------------------------------
    edges_diff = {}
    # See what edges we lost
    for key, val in edges_0.items():
        if key in edges_2:
            weight = edges_2[key] - edges_0[key]
            if weight < 0:
                # This edge gained flux
                color = "r"
            else:
                # This edge lost flux
                color = "b"
            edges_diff[key] = dict(color=color, weight=abs(weight))
        else:
            # This edge was lost in our predictions
            edges_diff[key] = dict(weight=val, color="r")

    # See what edges we gained.
    for key, val in edges_2.items():
        if key in edges_0:
            pass
        else:
            # This edge was gained in our predictions
            edges_diff[key] = dict(weight=val, color="b")

    # -----------------------------------------------
    # Calculate the change in flux at each node
    # -----------------------------------------------
    nodes_0 = dict([(i, G0.node[i]["flux"]) for i in G0.nodes()])
    nodes_2 = dict([(i, G2.node[i]["flux"]) for i in G2.nodes()])

    node_diff = {}
    for key, val in nodes_0.items():
        if key in nodes_2:
            diff = nodes_2[key] - val
            if diff > 0:
                color = "b"
            else:
                color = "r"
            node_diff[key] = dict(color=color, outer=nodes_2[key], inner=val)
        else:
            node_diff[key] = dict(color="r", outer=nodes_0[key], inner=0)

    for key, val in nodes_2.items():
        if key in nodes_0:
            pass
        else:
            node_diff[key] = dict(color="b", outer=val, inner=0)

    # -----------------------------------------------
    # Construct a network of differences
    # -----------------------------------------------
    Gdiff = nx.DiGraph()
    for key, val in edges_diff.items():
        Gdiff.add_edge(key[0],key[1],**val)

    for key, val in node_diff.items():
        Gdiff.node[key].update(**val)

    return G0, G2, Gdiff

if __name__ == "__main__":

    # Handle command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs=2, help="datasets to compare")
    parser.add_argument("dir", type=str, help="directory to run script")
    args = parser.parse_args()
    start = args.n[0]
    stop = args.n[1]

    main(start, stop, args.dir)
