import pickle
import numpy as np
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_adj_matrix(adj_file_path, weight_threshold=0.1):
    assert weight_threshold <= 1, "weight_threshold is large than 1.0"
    with open(adj_file_path, 'rb') as f:
        class_adj_matrix = pickle.load(f)
    class_adj_matrix = torch.tensor(class_adj_matrix >= weight_threshold, dtype=torch.float)
    print("class_adj_matrix:", class_adj_matrix)
    return class_adj_matrix


def generate_graph(class_adj_matrix, batch_labels):
    src_list, tgt_list = [], []
    relate_matrix = class_adj_matrix[batch_labels, batch_labels]
    print("relate_matrix:", relate_matrix)
    for row_idx in range(relate_matrix.shape[0]):
        for col_idx in range(row_idx, relate_matrix.shape[0]):
            if relate_matrix[row_idx, col_idx] == 1:
                src_list.append(row_idx)
                tgt_list.append(col_idx)
                src_list.append(col_idx)
                tgt_list.append(row_idx)
    src_list = torch.cat(src_list)
    tgt_list = torch.cat(tgt_list)
    edge_index = torch.stack([src_list, tgt_list])
    gt_node_labels = class_adj_matrix[batch_labels]
    print("edge_index.shape:", edge_index.shape, "gt_node_labels.shape:", gt_node_labels.shape)
    return edge_index, gt_node_labels


if __name__ == '__main__':
    tuberlin_adj_path = os.path.join(BASE_DIR, "Sketchy", "class_adj_byConceptNet5.5_.pkl")
    with open(tuberlin_adj_path, 'rb') as f:
        tuberlin_adj_matrix = pickle.load(f)
    # print("tuberlin_adj_matrix.shape:", tuberlin_adj_matrix.shape)  # sketchy_feature.shape: torch.Size([100, 654])
    # print("tuberlin_adj_matrix:", tuberlin_adj_matrix)  # sketchy_feature.shape: torch.Size([100, 654])
    count = 0
    for row_idx in range(tuberlin_adj_matrix.shape[0]):
        if tuberlin_adj_matrix[row_idx, row_idx] != 1.0:
            count += 1
            tuberlin_adj_matrix[row_idx, row_idx] = 1.0
    print("count:", count)
    tuberlin_relate_adj_path= os.path.join(BASE_DIR, "Sketchy", "ConceptNet5.5_relate_adj.pkl_emb")
    tuberlin_adj_matrix -=np.eye(tuberlin_adj_matrix.shape[0])
    with open(tuberlin_relate_adj_path, 'wb') as f:
        pickle.dump(tuberlin_adj_matrix, f)
    with open(tuberlin_relate_adj_path, 'rb') as f:
        tuberlin_adj_matrix = pickle.load(f)
    print("tuberlin_adj_matrix.shape:", tuberlin_adj_matrix.shape)  # sketchy_feature.shape: torch.Size([100, 654])
    print("tuberlin_adj_matrix:", tuberlin_adj_matrix)  # sketchy_feature.shape: torch.Size([100, 654])
