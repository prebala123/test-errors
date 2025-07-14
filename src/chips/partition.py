import numpy as np
import pymetis
import pickle

data_dir = '../../data/chips/NCSU-DigIC-GraphData-2023-07-25/xbar/'
clean_data_dir = '../../data/chips/clean_data/'
n_variants = 13

# Loop through all chips
for i in range(1, n_variants+1):
    connection_data = np.load(f'{data_dir}{i}/xbar_connectivity.npz')

    # Number of nodes and hyperedges
    num_nodes = max(connection_data['row']) + 1
    num_nets = max(connection_data['col']) + 1
    num_partitions = int(np.sqrt(num_nodes))

    # Convert hypergraph to bipartite graph representation in adjacency list
    adj_list = [[] for _ in range(num_nodes + num_nets)]
    for node, net in zip(connection_data['row'], connection_data['col']):
        adj_list[node].append(num_nodes + net)  
        adj_list[num_nodes + net].append(node)  

    # Step 2: Run Metis partitioning
    cuts, membership = pymetis.part_graph(num_partitions, adjacency=adj_list)
    arr = np.array(membership[:num_nodes])

    # Save partitioning data
    np.save(f'{clean_data_dir}{i}.partition.npy', arr)