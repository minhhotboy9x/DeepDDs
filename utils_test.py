import os
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from creat_data_DC import creat_data

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None):
        # xd: drug SMILES, xt: cell line, y: label, xt_featrue: cell line feature
        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xt_featrue, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        # print('cellId not found in cell_features:', cellId, cell_features)
        return False

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, xt_featrue, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.Tensor([labels]))
            cell = self.get_cell_feature(target, xt_featrue)

            if isinstance(cell, bool) and cell == False : # 如果读取cell失败则中断程序
                print('cell', cell)
                sys.exit()

            new_cell = []
            # print('cell_feature', cell_feature)
            for n in cell:
                new_cell.append(float(n))
            GCNData.cell = torch.FloatTensor([new_cell])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

class TestbedMutipleDataset(TestbedDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, 
                 xt=None, 
                 y=None, 
                 xt_featrue1=None, 
                 xt_featrue2=None, 
                 xt_featrue3=None, 
                 xt_featrue4=None, 
                 transform=None,
                 pre_transform=None, 
                 smile_graph=None):
        # xd: drug SMILES, xt: cell line, y: label, xt_featrue: cell line feature
        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, 
                         xt_featrue1,
                         xt_featrue2,
                         xt_featrue3,
                         xt_featrue4,
                         y, 
                         smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
    def get_multi_cell_feature(self, cellId, cell_features1, cell_features2, cell_features3, cell_features4):
        cell1 = self.get_cell_feature(cellId, cell_features1)
        cell2 = self.get_cell_feature(cellId, cell_features2)
        cell3 = self.get_cell_feature(cellId, cell_features3)
        cell4 = self.get_cell_feature(cellId, cell_features4)

        if isinstance(cell1, bool) and cell1 == False :
            print('cell1', cell1)
            sys.exit()
        if isinstance(cell2, bool) and cell2 == False :
            print('cell2', cell2)
            sys.exit()
        if isinstance(cell3, bool) and cell3 == False :
            print('cell3', cell3)
            sys.exit()
        if isinstance(cell4, bool) and cell4 == False :
            print('cell4', cell4)
            sys.exit()
        
        new_cell1 = np.array(cell1, dtype=float)
        new_cell2 = np.array(cell2, dtype=float)
        new_cell3 = np.array(cell3, dtype=float)
        new_cell4 = np.array(cell4, dtype=float)
        return new_cell1, new_cell2, new_cell3, new_cell4

    def process(self, 
                xd, 
                xt,
                xt_featrue1,
                xt_featrue2,
                xt_featrue3,
                xt_featrue4,
                y, 
                smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.Tensor([labels]))
            new_cell1, new_cell2, new_cell3, new_cell4 = \
                self.get_multi_cell_feature(target, xt_featrue1, xt_featrue2, xt_featrue3, xt_featrue4)

            GCNData.cell1 = torch.FloatTensor([new_cell1])
            GCNData.cell2 = torch.FloatTensor([new_cell2])
            GCNData.cell3 = torch.FloatTensor([new_cell3])
            GCNData.cell4 = torch.FloatTensor([new_cell4])
            
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

class TestbedEdgeDataset(TestbedDataset):
    def process(self, xd, xt, xt_featrue, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index, edge_attr = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            assert len(edge_index) == len(edge_attr), 'edge_index and edge_attr must be the same length!'
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                edge_attr=torch.Tensor(edge_attr),  
                                y=torch.Tensor([labels]))
            cell = self.get_cell_feature(target, xt_featrue)

            if isinstance(cell, bool) and cell == False : # 如果读取cell失败则中断程序
                print('cell', cell)
                sys.exit()

            new_cell = []
            # print('cell_feature', cell_feature)
            for n in cell:
                new_cell.append(float(n))
            GCNData.cell = torch.FloatTensor([new_cell])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci