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
        return output[:,0]

class Hypergraph:
    def __init__(self, incidence_matrix, driver_matrix, sink_matrix, vn_matrix):
        self.incidence_matrix = incidence_matrix
        self.driver_matrix = driver_matrix
        self.sink_matrix = sink_matrix
        self.vn_matrix = vn_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clean_data_dir = '../../data/chips/clean_data/'

n_samples = 13
train_idx = [1, 2, 3, 4, 5, 6, 7, 8]
train_data = []
valid_idx = [11]
valid_data = []
test_idx = [12]
test_data = []

for i in range(1, n_samples+1):
    connectivity = np.load(clean_data_dir + str(i) + '.connectivity.npz')
    incidence_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([connectivity['row'], connectivity['col']])), torch.ones(connectivity['dirs'].shape), dtype=torch.float).to(device)

    drivers = np.load(clean_data_dir + str(i) + '.drivers.npz')
    driver_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([drivers['col'], drivers['row']])), torch.ones(drivers['data'].shape), dtype=torch.float).to(device)

    sinks = np.load(clean_data_dir + str(i) + '.sinks.npz')
    sink_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([sinks['col'], sinks['row']])), torch.ones(sinks['data'].shape), dtype=torch.float).to(device)

    features = np.load(clean_data_dir + str(i) + '.features.npz')
    node_features = features['node_features']
    edge_features = features['net_features']
    wire = features['hpwl']

    node_features = torch.tensor(node_features, dtype=torch.float).to(device)
    edge_features = torch.tensor(edge_features, dtype=torch.float).to(device)
    wire = torch.tensor(wire, dtype=torch.float).to(device)

    num_nodes, num_node_features = node_features.shape
    num_edges, num_edge_features = edge_features.shape

    virtual_nodes = np.load(clean_data_dir + str(i) + '.virtual_nodes.npz')
    vn_rows = virtual_nodes['row']
    vn_cols = virtual_nodes['col']
    vn_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([vn_rows, vn_cols])), torch.ones(len(vn_rows)), dtype=torch.float).to(device)

    num_vn = vn_matrix.shape[0]
    num_vn_features = num_node_features
    vn_features = torch.zeros((num_vn, num_vn_features), dtype=torch.float).to(device)
    super_vn_features = torch.zeros(num_vn_features, dtype=torch.float).to(device)

    hypergraph = Hypergraph(incidence_matrix, driver_matrix, sink_matrix, vn_matrix)

    if i in train_idx:
        train_data.append((node_features, edge_features, vn_features, super_vn_features, hypergraph, wire))
    elif i in valid_idx:
        valid_data.append((node_features, edge_features, vn_features, super_vn_features, hypergraph, wire))
    elif i in test_idx:
        test_data.append((node_features, edge_features, vn_features, super_vn_features, hypergraph, wire))

# Initialize DE-HNN model
model = DEHNN(num_layers=4, node_in_features=num_node_features, edge_in_features=num_edge_features).to(device)
epochs = 100

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    for idx in range(len(train_data)):
        node_features, edge_features, vn_features, super_vn_features, hypergraph, congestion = train_data[idx]
        # Forward pass

        output = model(node_features, edge_features, vn_features, super_vn_features, hypergraph)
        
        target = congestion
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()

    # Print loss
    if epoch % 10 == 9:
        model.eval()
        node_features, edge_features, vn_features, super_vn_features, hypergraph, congestion = valid_data[0]
        output = model(node_features, edge_features, vn_features, super_vn_features, hypergraph)
        target = congestion
        valid_loss = criterion(output, target)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {valid_loss.item():.4f}')

torch.save(model, 'hpwl_xbar.pt')