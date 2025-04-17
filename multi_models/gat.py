import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
import pandas as pd
import numpy as np
from heatmap import get_map



# GAT  model
class MultiGATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, 
                 n_output=2, 
                 num_features_xt1=954, 
                 num_features_xt2=377, 
                 num_features_xt3=735, 
                 num_features_xt4=17737, 
                 output_dim=128, 
                 dropout=0.2, 
                 file=None):
        super(MultiGATNet, self).__init__()

        # graph drug layers
        self.drug1_gcn1 = GATConv(num_features_xd, output_dim, heads=10, dropout=dropout)
        self.drug1_gcn2 = GATConv(output_dim * 10, output_dim, dropout=dropout)
        # self.drug1_gcn3 = GATConv(output_dim, output_dim, dropout=dropout)
        self.drug1_fc_g1 = nn.Linear(output_dim, output_dim)
        # self.drug1_fc_g2 = nn.Linear(2048, output_dim)
        self.filename = file


        # DL cell featrues
        self.reduction1 = nn.Sequential(
            nn.Linear(num_features_xt1, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )

        self.reduction2 = nn.Sequential(
            nn.Linear(num_features_xt2, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )

        self.reduction3 = nn.Sequential(
            nn.Linear(num_features_xt3, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )

        self.reduction4 = nn.Sequential(
            nn.Linear(num_features_xt4, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )

        # 1D convolution on protein sequence
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        # self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        # self.fc_xt1 = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(output_dim * 6, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def get_col_index(self, x):
        row_size = len(x[:, 0])
        row = np.zeros(row_size)
        col_size = len(x[0, :])
        for i in range(col_size):
            row[np.argmax(x[:, i])] += 1
        return row

    def save_num(self, d, path):
        d = d.cpu().numpy()
        ind = self.get_col_index(d)
        ind = pd.DataFrame(ind)
        ind.to_csv('data/case_study/' + path + '_index.csv', header=0, index=0)
        # 下面是load操作
        # read_dictionary = np.load('my_file.npy').item()
        # d = pd.DataFrame(d)
        # d.to_csv('data/result/' + path + '.csv', header=0, index=0)

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell1, cell2, cell3, cell4 = \
            data1.x, data1.edge_index, data1.batch, data1.cell1, data1.cell2, data1.cell3, data1.cell4

        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        # begin_x1 = np.array(x1.cpu().detach().numpy())
        x1 = self.drug1_gcn1(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.drug1_gcn2(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = gmp(x1, batch1)         # global max pooling


        x1 = self.drug1_fc_g1(x1)
        x1 = self.relu(x1)

        # deal drug2
        # begin_x2 = np.array(x2.cpu().detach().numpy())
        x2 = self.drug1_gcn1(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x2 = self.drug1_gcn2(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x2 = gmp(x2, batch2)  # global max pooling


        x2 = self.drug1_fc_g1(x2)
        x2 = self.relu(x2)

        # deal cell
        cell1 = F.normalize(cell1, 2, 1)
        cell2 = F.normalize(cell2, 2, 1)
        cell3 = F.normalize(cell3, 2, 1)
        cell4 = F.normalize(cell4, 2, 1)
        cell_vector1 = self.reduction1(cell1)
        cell_vector2 = self.reduction2(cell2)
        cell_vector3 = self.reduction3(cell3)
        cell_vector4 = self.reduction4(cell4)



        # concat
        xc = torch.cat((x1, x2, cell_vector1, cell_vector2, cell_vector3, cell_vector4), 1)
        xc = F.normalize(xc, 2, 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
