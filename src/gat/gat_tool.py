import pickle

import torch


def get_adj_matrix(adj_file_path,word_vec_path, weight_threshold=0.1):
    assert weight_threshold <= 1, "weight_threshold is large than 1.0"
    with open(adj_file_path, 'rb') as f:
        class_adj_matrix = pickle.load(f)
    class_adj_matrix = torch.tensor(class_adj_matrix >= weight_threshold, dtype=torch.float)
    # print("class_adj_matrix:", class_adj_matrix)
    with open(word_vec_path, 'rb') as f:
        word_vec_matrix = pickle.load(f)
    word_vec_matrix = torch.tensor(word_vec_matrix, dtype=torch.float)
    return class_adj_matrix,word_vec_matrix


def generate_graph(class_adj_matrix,word_vec_matrix, batch_labels):
    src_list, tgt_list = [], []
    relate_matrix = class_adj_matrix[batch_labels].index_select(-1, batch_labels)
    # print("relate_matrix:", relate_matrix.shape, relate_matrix)
    for row_idx in range(relate_matrix.shape[0]):
        for col_idx in range(row_idx, relate_matrix.shape[0]):
            if relate_matrix[row_idx, col_idx] == 1:
                src_list.append(row_idx)
                tgt_list.append(col_idx)
                if row_idx != col_idx:
                    src_list.append(col_idx)
                    tgt_list.append(row_idx)
    src_list = torch.cat([torch.tensor(src_list)])
    tgt_list = torch.cat([torch.tensor(tgt_list)])
    edge_index = torch.stack([src_list, tgt_list])
    word_vectors = word_vec_matrix[batch_labels]
    # print("edge_index.shape:", edge_index.shape, "gt_node_labels.shape:", gt_node_labels.shape)
    # print("edge_index:", edge_index)
    # print("gt_node_labels:", gt_node_labels)
    return edge_index, word_vectors
