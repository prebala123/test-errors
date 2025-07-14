import numpy as np
import torch
import torch.nn as nn
import pickle

class DEHNNLayer(nn.Module):
    def __init__(self, node_in_features, edge_in_features, vn_features, hidden_features):
        super(DEHNNLayer, self).__init__()
        self.node_mlp1 = nn.Sequential(nn.Linear(edge_in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, edge_in_features))
        
        self.edge_mlp2 = nn.Sequential(nn.Linear(node_in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, node_in_features))
        
        self.edge_mlp3 = nn.Sequential(nn.Linear(2 * node_in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, 2 * node_in_features))

        self.node_to_virtual_mlp = nn.Sequential(nn.Linear(node_in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, vn_features))
        
        self.virtual_to_higher_virtual_mlp = nn.Sequential(nn.Linear(vn_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, vn_features))
        
        self.higher_virtual_to_virtual_mlp = nn.Sequential(nn.Linear(vn_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, vn_features))
        
        self.virtual_to_node_mlp = nn.Sequential(nn.Linear(vn_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, edge_in_features))


    def forward(self, node_features, edge_features, vn_features, super_vn_features, hypergraph):

        # Node Update
        transformed_edge_features = self.node_mlp1(edge_features)
        updated_node_features = torch.matmul(hypergraph.incidence_matrix, transformed_edge_features)

        # Edge Update
        transformed_node_features = self.edge_mlp2(node_features)
        driver_features = torch.matmul(hypergraph.driver_matrix, transformed_node_features)
        sink_features = torch.matmul(hypergraph.sink_matrix, transformed_node_features)
        updated_edge_features = torch.cat([driver_features, sink_features], dim=1)
        updated_edge_features = self.edge_mlp3(updated_edge_features)
        
        # First Level VN Update
        node_to_virtual_features = self.node_to_virtual_mlp(node_features)
        updated_vn_features = torch.matmul(hypergraph.vn_matrix, node_to_virtual_features)
        updated_vn_features += self.higher_virtual_to_virtual_mlp(super_vn_features)

        # Top Level VN Update
        virtual_to_higher_virtual_features = self.virtual_to_higher_virtual_mlp(vn_features)
        updated_super_vn_features = torch.sum(virtual_to_higher_virtual_features, dim=0)

        # VN to node update
        virtual_to_node_features = self.virtual_to_node_mlp(vn_features)
        propagated_features = torch.matmul(hypergraph.vn_matrix.T, virtual_to_node_features)
        updated_node_features += propagated_features

        return updated_node_features, updated_edge_features, updated_vn_features, updated_super_vn_features


class DEHNN(nn.Module):
    def __init__(self, num_layers, node_in_features, edge_in_features, hidden_features=24):
        super(DEHNN, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Create multiple layers for DEHNN
        vn_in_features = node_in_features
        for i in range(num_layers):
            self.layers.append(DEHNNLayer(node_in_features, edge_in_features, vn_in_features, hidden_features))
            node_in_features, edge_in_features = edge_in_features, node_in_features
            edge_in_features *= 2
        
        self.output_layer = nn.Sequential(nn.Linear(edge_in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, 1))

    def forward(self, node_features, edge_features, vn_features, super_vn_features, hypergraph):
        # Pass through each layer
        for layer in self.layers:
            node_features, edge_features, vn_features, super_vn_features = layer(node_features, edge_features, vn_features, super_vn_features, hypergraph)
        
        # Output prediction for nodes
        output = self.output_layer(edge_features)
        return output

class Hypergraph:
    def __init__(self, incidence_matrix, driver_matrix, sink_matrix, vn_matrix):
        self.incidence_matrix = incidence_matrix
        self.driver_matrix = driver_matrix
        self.sink_matrix = sink_matrix
        self.vn_matrix = vn_matrix

def open_chip(i):
    path = '../../data/chips/2023-03-06_data/'

    with open(path + f'{i}.bipartite.pkl', 'rb') as f:
        bipartite = pickle.load(f)

    with open(path + f'{i}.degree.pkl', 'rb') as f:
        degree = pickle.load(f)

    with open(path + f'{i}.eigen.10.pkl', 'rb') as f:
        eigen = pickle.load(f)

    with open(path + f'{i}.global_information.pkl', 'rb') as f:
        global_info = pickle.load(f)

    with open(path + f'{i}.metis_part_dict.pkl', 'rb') as f:
        metis = pickle.load(f)

    with open(path + f'{i}.net_demand_capacity.pkl', 'rb') as f:
        net_demand_capacity = pickle.load(f)

    with open(path + f'{i}.net_features.pkl', 'rb') as f:
        net_feats = pickle.load(f)

    with open(path + f'{i}.net_hpwl.pkl', 'rb') as f:
        hpwl = pickle.load(f)

    with open(path + f'{i}.nn_conn.pkl', 'rb') as f:
        nn_conn = pickle.load(f)

    with open(path + f'{i}.node_features.pkl', 'rb') as f:
        node_feats = pickle.load(f)

    with open(path + f'{i}.pl_fix_part_dict.pkl', 'rb') as f:
        pl = pickle.load(f)

    with open(path + f'{i}.star.pkl', 'rb') as f:
        star = pickle.load(f)

    with open(path + f'{i}.targets.pkl', 'rb') as f:
        targets = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    incidence_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([bipartite['instance_idx'], bipartite['net_idx']])), torch.ones(bipartite['edge_dir'].shape), dtype=torch.float).to(device)

    driver_idx = bipartite['edge_dir'] == 1
    driver_row = bipartite['instance_idx'][driver_idx]
    driver_col = bipartite['net_idx'][driver_idx]
    driver_dir = bipartite['edge_dir'][driver_idx]
    driver_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([driver_col, driver_row])), torch.ones(driver_dir.shape), dtype=torch.float).to(device)

    sink_idx = bipartite['edge_dir'] == 0
    sink_row = bipartite['instance_idx'][sink_idx]
    sink_col = bipartite['net_idx'][sink_idx]
    sink_dir = bipartite['edge_dir'][sink_idx]
    sink_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([sink_col, sink_row])), torch.ones(sink_dir.shape), dtype=torch.float).to(device)

    num_nodes = node_feats['num_instances']
    node_features = node_feats['instance_features']
    node_features = torch.tensor(node_features, dtype=torch.float).to(device)
    evects = torch.tensor(eigen['evects'][:num_nodes]).to(device)
    node_features = torch.cat([node_features, evects], dim=1)

    net_features = net_feats['instance_features'][:,:10]
    edge_features = torch.tensor(net_features, dtype=torch.float).to(device)

    num_nodes, num_node_features = node_features.shape
    num_edges, num_edge_features = edge_features.shape
    demand = targets['demand'].reshape(num_nodes,1)
    demand = torch.tensor(demand, dtype=torch.float).to(device)
    wire = hpwl['hpwl'].reshape(num_edges,1)
    wire = torch.tensor(wire, dtype=torch.float).to(device)

    vn_row = []
    vn_col = []

    for k, v in metis.items():
        if k < num_nodes:
            vn_row.append(v)
            vn_col.append(k)

    vn_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([vn_row, vn_col])), torch.ones(len(vn_row)), dtype=torch.float).to(device)

    num_vn = vn_matrix.shape[0]
    num_vn_features = num_node_features
    vn_features = torch.zeros((num_vn, num_vn_features), dtype=torch.float).to(device)
    super_vn_features = torch.zeros(num_vn_features, dtype=torch.float).to(device)

    hypergraph = Hypergraph(incidence_matrix, driver_matrix, sink_matrix, vn_matrix)

    return node_features, edge_features, vn_features, super_vn_features, hypergraph, wire

num_node_features = 16
num_edge_features = 10

device = 'cuda'

train_idx = [16, 39, 40, 62, 68]
valid_idx = 44

valid_node_features, valid_edge_features, valid_vn_features, valid_super_vn_features, valid_hypergraph, valid_wire = open_chip(valid_idx)

# Initialize DE-HNN model
model = DEHNN(num_layers=4, node_in_features=num_node_features, edge_in_features=num_edge_features).to(device)
epochs = 100

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    for i in train_idx:
        node_features, edge_features, vn_features, super_vn_features, hypergraph, wire = open_chip(i)

        output = model(node_features, edge_features, vn_features, super_vn_features, hypergraph)

        loss = criterion(output, wire)
        
        # Backward pass
        loss.backward()
        optimizer.step()

    # Print loss
    if epoch % 10 == 9:
        model.eval()
        output = model(valid_node_features, valid_edge_features, valid_vn_features, valid_super_vn_features, valid_hypergraph)
        valid_loss = criterion(output, valid_wire)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {valid_loss.item():.4f}')

torch.save(model, 'hpwl_superblue.pt')