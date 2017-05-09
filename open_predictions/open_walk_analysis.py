__doc__ = """
# Probability flux through node *j*

$$
F_j = \frac{\pi_{i \rightarrow j}}{Z} \cdot F_i
$$

$$
Z = \sum_{k < i} \pi_{i \rightarrow k}
$$
"""

# -------------------------------------------------
# Graphing/plotting code
# -------------------------------------------------  

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from latticeproteins.sequences import hamming_distance

def get_source_node(G):
    x = set([i for i, j in G.edges()])
    y = set([j for i, j in G.edges()])
    out = x.difference(y)
    if len(out) == 1:
        return list(out)[0]
    else:
        raise Exception("Could not resolve source node.")

def flux_out_of_node(G, node_i):
    # Get flux coming from source
    total_flux_avail = G.node[node_i]["flux"]
    edges = {}
    # Normalize the transition probability from source
    norm = sum([G.edge[node_i][node_j]["weight"] for node_j in G.neighbors(node_i)])
    # Iterate over neighbors divvy up flux across neighbors
    for node_j in G.neighbors(node_i):
        fixation = G.edge[node_i][node_j]["weight"]
        dflux = (fixation/norm) * total_flux_avail
        if dflux > 0.01:
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
        levels[key] = set(val)
    return levels

def radial(r, theta):
    return (r*np.cos(theta), r*np.sin(theta))

def ring_position(G, root):
    levels = ring_levels(G, root)
    pos = {}
    for i in range(len(levels)):
        nodes = levels[i]
        for j, node in enumerate(nodes):
            angle = 2*np.pi / len(nodes)
            pos[node] = radial(i, j*angle)
    return pos

def build_graphs(edges1, edges2):
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

    seq = get_source_node(G0)
    # -----------------------------------------------
    # Calculate the flux at each node and edge
    # -----------------------------------------------
    G0 = flux_from_source(G0, seq)
    G2 = flux_from_source(G2, seq)

    # Remove nodes that have small flux
    nodes_to_remove = []
    for node in G0.nodes():
        if G0.node[node]["flux"] < 0.001:
            nodes_to_remove.append(node)

    nodes_to_remove2 = []
    for node in G2.nodes():
        if G2.node[node]["flux"] < 0.001:
            nodes_to_remove2.append(node)

    G0.remove_nodes_from(nodes_to_remove)
    G2.remove_nodes_from(nodes_to_remove2)

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

def plot_networks(G1, G2, Gdiff):
    """"""
    # options
    node_scale = 600
    edge_scale = 75
    node_color = "k"
    
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle
    
    def draw_circles(ax):
        """Draw circles add increasing hamming distances for each network."""
        for i in range(0,7):
            circle = Circle((0, 0), i, facecolor='none',
                    edgecolor="k", linewidth=.5, alpha=0.5, linestyle="--")
            ax.add_patch(circle)

    
    # Initialize a figure
    fig = plt.figure(figsize=(10,30))
    
    # Initialize a gridspec
    gs = GridSpec(3, 1)
       
    seq = get_source_node(G1)
    # Calculate the positions for all nodes on rings
    pos = ring_position(Gdiff, seq)

    # -------------------------------------------------
    # Draw the first network
    # -------------------------------------------------
    
    ax1 = plt.subplot(gs[0, 0])
    
    # Set the widths of the edges to the delta flux attribute of each edge.
    edge_widths = np.array([G1.edge[i][j]["delta_flux"] for i,j in G1.edges()])
    edge_widths = edge_widths * edge_scale
    #edge_widths = np.ma.log10(edge_widths).filled(0) * edge_scale
    nx.draw_networkx_edges(G1, pos=pos, ax=ax1,
        width=edge_widths,                
        arrows=False,
        edge_color="gray",
        alpha=0.5
    )
    
    # Set the node sizes to the amount of flux passing through each node.
    node_size = [G1.node[i]["flux"] * node_scale for i in G1.nodes()]
    #node_size = np.ma.log10(node_size).filled(0)

    nx.draw_networkx_nodes(G1, pos=pos, ax=ax1,
        node_size=node_size,                
        linewidths=None,
        node_color=node_color
    )
    
    bad_nodes1 = [node for node in Gdiff.nodes() if node not in G1.nodes()]

    nx.draw_networkx_nodes(Gdiff, pos=pos, ax=ax1,
        nodelist = bad_nodes1,
        node_shape = "x",
        node_size = node_scale*.25,
        linewidths = None,
        node_color = "m"
    )
    
    # Draw circles
    draw_circles(ax1)
    ax1.axis("equal")
    ax1.axis("off")
    
    # -------------------------------------------------
    # Draw the second network
    # -------------------------------------------------
    
    ax2 = plt.subplot(gs[1, 0])

    
    # Set the widths of the edges to the delta flux attribute of each edge.
    edge_widths = np.array([G2.edge[i][j]["delta_flux"] for i,j in G2.edges()])
    edge_widths = edge_widths * edge_scale
    #edge_widths = np.ma.log10(edge_widths).filled(0) * edge_scale
    nx.draw_networkx_edges(G2, pos=pos, ax=ax2,
        width=edge_widths,                
        arrows=False,
        edge_color="gray",
        alpha=0.5
    )
    
    # Set the node sizes to the amount of flux passing through each node.
    node_size = [G2.node[i]["flux"] * node_scale for i in G2.nodes()]
    #node_size = np.ma.log10(node_size).filled(0)

    nx.draw_networkx_nodes(G2, pos=pos, ax=ax2,
        node_size=node_size,                
        linewidths=None,
        node_color=node_color
    )

    bad_nodes2 = [node for node in Gdiff.nodes() if node not in G2.nodes()]

    nx.draw_networkx_nodes(Gdiff, pos=pos, ax=ax2,
        nodelist = bad_nodes2,
        node_shape = "x",
        node_size = node_scale*.25,
        linewidths = None,
        node_color = "m"
    )
        
    # Draw circles
    draw_circles(ax2) 
    ax2.axis("equal")
    ax2.axis("off")
    
    # -------------------------------------------------
    # Draw difference network
    # -------------------------------------------------
        
    ax3 = plt.subplot(gs[2, 0])

    
    # Set the widths of the edges to the delta flux attribute of each edge.
    edge_widths = np.array([Gdiff.edge[i][j]["weight"] for i,j in Gdiff.edges()])
    edge_widths = edge_widths * edge_scale
    #edge_widths = np.ma.log10(edge_widths).filled(0) * edge_scale 
    edge_color = [Gdiff.edge[i][j]["color"] for i,j in Gdiff.edges()]

    nx.draw_networkx_edges(Gdiff, pos=pos, ax=ax3,
        width=edge_widths,                
        arrows=False,
        edge_color=edge_color,
        alpha=0.5
    )
    
    # Set the node sizes to the amount of flux passing through each node.
    node_size = [Gdiff.node[i]["outer"] * node_scale for i in Gdiff.nodes()]
    node_color = [Gdiff.node[i]["color"]  for i in Gdiff.nodes()]
    nx.draw_networkx_nodes(Gdiff, pos=pos, ax=ax3,
        node_size=node_size,                
        linewidths=None,
        node_color=node_color
    )

    # Set the node sizes to the amount of flux passing through each node.
    node_size = [Gdiff.node[i]["inner"] * node_scale for i in Gdiff.nodes()]
    nx.draw_networkx_nodes(Gdiff, pos=pos, ax=ax3,
        node_size=node_size,                
        linewidths=None,
        node_color="w"
    )    
    
    # Draw circles
    draw_circles(ax3) 

    ax3.axis("equal")
    ax3.axis("off")
    return fig