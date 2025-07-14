# Import Statements
import numpy as np
import json
import gzip
from scipy.stats import binned_statistic_2d
from collections import defaultdict
import torch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh

# Data Locations
raw_data_dir = '../../data/chips/NCSU-DigIC-GraphData-2023-07-25/'
clean_data_dir = '../../data/chips/clean_data/'
design = 'xbar'
n_variants = 13

# Create file path for each chip in dataset
sample_names = []
corresponding_design = []
corresponding_variant = []
for idx in range(n_variants):
    sample_name = raw_data_dir + design + '/' + str(idx + 1) + '/'
    sample_names.append(sample_name)
    corresponding_design.append(design)
    corresponding_variant.append(idx + 1)

# +------------------------------+
# | Information about cell types |
# +------------------------------+

# Open cells dataset
cells_fn = raw_data_dir + 'cells.json.gz'
with gzip.open(cells_fn, 'r') as fin:
    cell_data = json.load(fin)

# Get widths and heights for all cell types
widths = []
heights = []
for idx in range(len(cell_data)):
    width = cell_data[idx]['width']
    height = cell_data[idx]['height']
    widths.append(width)
    heights.append(height)

widths = np.array(widths)
heights = np.array(heights)

min_cell_width = np.min(widths)
max_cell_width = np.max(widths)
min_cell_height = np.min(heights)
max_cell_height = np.max(heights)

# Scale all widths and heights of each cell type from 0 to 1
widths = (widths - min_cell_width) / (max_cell_width - min_cell_width)
heights = (heights - min_cell_height) / (max_cell_height - min_cell_height)

# For each cell type, map which pins are inputs and outputs
cell_to_edge_dict = {item['id']:{inner_item['id']: inner_item['dir'] for inner_item in item['terms']} for item in cell_data}

# Loop through all chips in the dataset and create features for them
for sample in range(n_variants):

    # +--------------------------------+
    # | Information about current chip |
    # +--------------------------------+

    # Navigate to folder with chip information
    folder = sample_names[sample]
    design = corresponding_design[sample]
    instances_nets_fn = folder + design + '.json.gz'

    print('--------------------------------------------------')
    print('Folder:', folder)
    print('Design:', design)
    print('Variant:', (sample+1))

    # Open file with information about the cells and nets
    with gzip.open(instances_nets_fn, 'r') as fin:
        instances_nets_data = json.load(fin)

    instances = instances_nets_data['instances']
    nets = instances_nets_data['nets']

    # Map each cell to its type
    inst_to_cell = {item['id']:item['cell'] for item in instances}

    num_instances = len(instances)
    num_nets = len(nets)

    print('Number of instances:', num_instances)
    print('Number of nets:', num_nets)

    # +---------------------------+
    # | Instance feature creation |
    # +---------------------------+

    # Features are x location, y location, cell type, width, height, and orientation
    # x and y location will later be removed because we don't have placement information at prediction time
    xloc_list = [instances[idx]['xloc'] for idx in range(num_instances)]
    yloc_list = [instances[idx]['yloc'] for idx in range(num_instances)]
    cell = [instances[idx]['cell'] for idx in range(num_instances)]
    cell_width = [widths[cell[idx]] for idx in range(num_instances)]
    cell_height = [heights[cell[idx]] for idx in range(num_instances)]
    orient = [instances[idx]['orient'] for idx in range(num_instances)]
    
    x_min = min(xloc_list)
    x_max = max(xloc_list)
    y_min = min(yloc_list)
    y_max = max(yloc_list)

    # Scale widths and heights between 0 and 1
    X = np.expand_dims(np.array(xloc_list), axis = 1)
    Y = np.expand_dims(np.array(yloc_list), axis = 1)
    X = (X - x_min) / (x_max - x_min)
    Y = (Y - y_min) / (y_max - y_min)

    cell = np.expand_dims(np.array(cell), axis = 1)
    cell_width = np.expand_dims(np.array(cell_width), axis = 1)
    cell_height = np.expand_dims(np.array(cell_height), axis = 1)
    orient = np.expand_dims(np.array(orient), axis = 1)

    # Create an array with all instance features
    instance_features = np.concatenate((X, Y, cell, cell_width, cell_height, orient), axis = 1)

    # +-------------------+
    # | Connectivity data |
    # +-------------------+

    # Open connectivity data
    connection_fn = folder + design + '_connectivity.npz'
    connection_data = np.load(connection_fn)
    
    # Get the direction of each edge to tell if cell is driver or sink of net
    dirs = []
    edge_t = connection_data['data']
    instance_idx = connection_data['row']
    
    for idx in range(len(instance_idx)):
        inst = instance_idx[idx]
        cell = inst_to_cell[inst]
        edge_dict = cell_to_edge_dict[cell]
        t = edge_t[idx]
        direction = edge_dict[t]
        dirs.append(direction)

    dirs = np.array(dirs)

    # Map the drivers and sinks of every net
    driver_sink_map = defaultdict(lambda: (None, []))

    # Extract unique nodes and edges
    nodes = list(set(connection_data['row']))
    edges = list(set(connection_data['col']))

    # Map each driver to all of its sinks for each net
    for node, edge, direction in zip(connection_data['row'], connection_data['col'], dirs):
        if direction == 1:  # Driver
            driver_sink_map[edge] = (node, driver_sink_map[edge][1])
        elif direction == 0:  # Sink
            driver_sink_map[edge][1].append(node)

    driver_sink_map = dict(driver_sink_map)

    # +----------------------+
    # | Net feature creation |
    # +----------------------+

    # Determine degree of each net
    net_features = {}
    for k, v in driver_sink_map.items():
        if v[0]:
            net_features[k] = [len(v[1]) + 1]
        else:
            net_features[k] = [len(v[1])]

    # Half perimeter wire length
    wire_length = []
    for k, v in driver_sink_map.items():
        minx, miny, maxx, maxy = float('inf'), float('inf'), -float('inf'), -float('inf')
        if v[0]:
            minx = min(minx, xloc_list[v[0]])
            miny = min(miny, yloc_list[v[0]])
            maxx = max(maxx, xloc_list[v[0]])
            maxy = max(maxy, yloc_list[v[0]])
        for i in v[1]:
            minx = min(minx, xloc_list[i])
            miny = min(miny, yloc_list[i])
            maxx = max(maxx, xloc_list[i])
            maxy = max(maxy, yloc_list[i])
        wire = (maxx - minx) + (maxy - miny)
        wire_length.append(wire)
    wire_length = np.array(wire_length)
    wire_length = np.log2(wire_length)
    wire_length = np.clip(wire_length, a_min=0, a_max=None)

    # +-----------------------+
    # | Positional Embeddings |
    # +-----------------------+

    # Use laplacian eigenmaps to get positional embeddings of each node and net
    instance_idx = connection_data['row']
    net_idx = connection_data['col']
    net_idx += num_instances

    # Create bipartite graph of nodes and nets
    v1 = torch.unsqueeze(torch.Tensor(np.concatenate([instance_idx, net_idx], axis = 0)).long(), dim = 1)
    v2 = torch.unsqueeze(torch.Tensor(np.concatenate([net_idx, instance_idx], axis = 0)).long(), dim = 1)
    undir_edge_index = torch.transpose(torch.cat([v1, v2], dim = 1), 0, 1)

    # Get laplacian matrix and eigenvectors for embeddings
    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization = "sym", num_nodes = num_instances + num_nets)
    )
    evals, evects = eigsh(L, k = 10, which='SM')

    # Create node features with previous instance features except real positional data, and new positional encodings
    node_features = {}
    for i in range(num_instances):
        node_features[i] = np.concatenate([instance_features[i, 2:], evects[i]])

    node_features = np.array(list(node_features.values()))
    net_features = np.array(list(net_features.values()))
    
    # +---------------------+
    # | Get congestion data |
    # +---------------------+

    # Open congestion data file
    congestion_fn = folder + design + '_congestion.npz'
    congestion_data = np.load(congestion_fn)

    congestion_data_demand = congestion_data['demand']
    congestion_data_capacity = congestion_data['capacity']

    num_layers = len(list(congestion_data['layerList']))

    ybl = congestion_data['yBoundaryList']
    xbl = congestion_data['xBoundaryList']

    all_demand = []
    all_capacity = []

    # Loop through every wiring layer
    for layer in list(congestion_data['layerList']):
        # Get capacity and demand for each cell
        lyr = list(congestion_data['layerList']).index(layer)

        ret = binned_statistic_2d(xloc_list, yloc_list, None, 'count', bins = [xbl[1:], ybl[1:]], expand_binnumbers = True)

        i_list = np.array([ret.binnumber[0, idx] - 1 for idx in range(num_instances)])
        j_list = np.array([ret.binnumber[1, idx] - 1 for idx in range(num_instances)])

        demand_list = congestion_data_demand[lyr, i_list, j_list].flatten()
        capacity_list = congestion_data_capacity[lyr, i_list, j_list].flatten()

        demand_list = np.array(demand_list)
        capacity_list = np.array(capacity_list)

        all_demand.append(np.expand_dims(demand_list, axis = 1))
        all_capacity.append(np.expand_dims(capacity_list, axis = 1))

        average_demand = np.mean(demand_list)
        average_capacity = np.mean(capacity_list)
        average_diff = np.mean(capacity_list - demand_list)
        count_congestions = np.sum(demand_list > capacity_list)

    demand = np.concatenate(all_demand, axis = 1).sum(axis=1)
    capacity = np.concatenate(all_capacity, axis = 1).sum(axis=1)

    # Create congestion map
    congestion = demand / capacity
    congestion = (congestion >= 0.9).astype(int)
    actual_congestion = []
    for con in congestion:
        if con == 0:
            actual_congestion.append([1, 0])
        else:
            actual_congestion.append([0, 1])
    actual_congestion = np.array(actual_congestion)

    # Create mapping from nodes to virtual nodes
    partition = np.load(f'{clean_data_dir}{sample+1}.partition.npy')
    node_to_virtual = {i: p for i, p in enumerate(partition)}
    virtual_to_node = defaultdict(list)
    for i, p in enumerate(partition):
        virtual_to_node[p].append(i)

    vn_rows = []
    vn_cols = []

    for k, v in virtual_to_node.items():
        for n in v:
            vn_rows.append(k)
            vn_cols.append(n)

    vn_rows = np.array(vn_rows)
    vn_cols = np.array(vn_cols)

    # Create mappings for drivers
    idx = dirs == 1
    driver_rows = connection_data['row'][idx]
    driver_cols = connection_data['col'][idx]
    driver_data = connection_data['data'][idx]

    # Create mappings for sinks
    idx = dirs == 0
    sink_rows = connection_data['row'][idx]
    sink_cols = connection_data['col'][idx]
    sink_data = connection_data['data'][idx]

    # +-----------------------+
    # | Save all new features |
    # +-----------------------+


    np.savez(f'{clean_data_dir}{sample+1}.connectivity.npz', row=connection_data['row'], col=connection_data['col'], dirs=dirs)
    np.savez(f'{clean_data_dir}{sample+1}.drivers.npz', row=driver_rows, col=driver_cols, data=driver_data)
    np.savez(f'{clean_data_dir}{sample+1}.sinks.npz', row=sink_rows, col=sink_cols, data=sink_data)
    np.savez(f'{clean_data_dir}{sample+1}.features.npz', node_features=node_features, net_features=net_features, congestion=actual_congestion, demand=demand, hpwl=wire_length)
    np.savez(f'{clean_data_dir}{sample+1}.virtual_nodes.npz', row=vn_rows, col=vn_cols)
