import csv
from itertools import islice

import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
# from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils_test import *



def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1: ]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index



def get_mutil_cell_feats(cellfile1,
                        cellfile2,
                        cellfile3,
                        cellfile4):
    cell_feature1 = []
    cell_feature2 = []
    cell_feature3 = []
    cell_feature4 = []


    with open(cellfile1) as csvfile1:
        csv_reader = csv.reader(csvfile1)  
        for row in csv_reader:
            cell_feature1.append(row)
    cell_feature1 = np.array(cell_feature1)

    with open(cellfile2) as csvfile2:
        csv_reader = csv.reader(csvfile2)
        for row in csv_reader:
            cell_feature2.append(row)
    cell_feature2 = np.array(cell_feature2)

    with open(cellfile3) as csvfile3:
        csv_reader = csv.reader(csvfile3)
        for row in csv_reader:
            cell_feature3.append(row)
    cell_feature3 = np.array(cell_feature3)

    with open(cellfile4) as csvfile4:
        csv_reader = csv.reader(csvfile4)
        for row in csv_reader:
            cell_feature4.append(row)
    cell_feature4 = np.array(cell_feature4)
    
    return cell_feature1, cell_feature2, cell_feature3, cell_feature4

def create_multi_cell_data(
        datafile,
        cellfile1,
        cellfile2,
        cellfile3,
        cellfile4):
    
    cell_feature1, cell_feature2, cell_feature3, cell_feature4 = get_mutil_cell_feats(
        cellfile1, cellfile2, cellfile3, cellfile4)
    print('cell_features1', cell_feature1)
    print('cell_features2', cell_feature2)
    print('cell_features3', cell_feature3)
    print('cell_features4', cell_feature4)


    compound_iso_smiles = []
    df = pd.read_csv('data\smiles.csv')
    compound_iso_smiles += list(df['smile'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    print('compound_iso_smiles', compound_iso_smiles)
    for smile in compound_iso_smiles:
        print('smiles', smile)
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    datasets = datafile
    # convert to PyTorch data format
    processed_data_file_train = 'data/processed/' + datasets + '_train.pt'
    if ((not os.path.isfile(processed_data_file_train))):
        df = pd.read_csv('data/' + datasets + '.csv')
        drug1, drug2, cell, label = list(df['drug1']), list(df['drug2']), list(df['cell']), list(df['label'])
        drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)
        print('开始创建数据')
        TestbedMutipleDataset(root='data', dataset=datafile + '_drug1', xd=drug1, xt=cell, 
                            xt_featrue1=cell_feature1, 
                            xt_featrue2=cell_feature2, 
                            xt_featrue3=cell_feature3, 
                            xt_featrue4=cell_feature4,  
                            y=label,
                            smile_graph=smile_graph)
        
        TestbedMutipleDataset(root='data', dataset=datafile + '_drug2', xd=drug1, xt=cell, 
                            xt_featrue1=cell_feature1, 
                            xt_featrue2=cell_feature2, 
                            xt_featrue3=cell_feature3, 
                            xt_featrue4=cell_feature4,  
                            y=label,
                            smile_graph=smile_graph)
        
        print('创建数据成功')
        print('preparing ', datasets + '_.pt in pytorch format!')

if __name__ == "__main__":
    cellfile1 = 'data/DRP_merged/processed_independent_cell_features_filtered.csv'
    cellfile2 = 'data/DRP_merged/genetic_feature_matrix_meth_filtered.csv'
    cellfile3 = 'data/DRP_merged/genetic_feature_matrix_pan_filtered.csv'
    cellfile4 = 'data/DRP_merged/processed_cell_line_RMA_filtered.csv'
    
    da = ['new_labels_0_10_drp_filtered']
    for datafile in da:
        create_multi_cell_data(datafile, 
                               cellfile1, 
                               cellfile2,
                               cellfile3,
                               cellfile4)
