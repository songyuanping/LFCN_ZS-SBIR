import pickle
import numpy as np
import torch
import os
import requests
import threading,threadpoolctl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_word_matrix(path, name_id_txt):
    # path = r"data/Sketchy/word2vec-google-news.npy"
    # data = np.load(path, allow_pickle=True).item()
    # print(type(data))
    # print(len(data.keys()), data.keys())
    # print(data["pig"].shape)

    # path = os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "word2vec-google-news.npy")
    data = np.load(path, allow_pickle=True).item()
    print(type(data))
    print(len(data.keys()), data.keys())
    # print(data["apple"].shape)

    cnames = []
    with open(name_id_txt) as f:
        lines = f.readlines()
    count = 0
    word_matrix = []
    for line in lines:
        items = line.strip().split()
        # print("\nitems:", items)
        c_name = "_".join(items[:-1]).replace("-", "_")

        if c_name in data.keys():
            count += 1
            # print("c_name: ", c_name, data[c_name].shape)
            word_matrix.append(data[c_name])
            cnames.append(c_name)
        else:
            for key in data.keys():
                new_name = str(key).replace("_", "").replace("(", "").replace(")", "")
                # print("c_name:", c_name, "new_name:", new_name)
                if new_name == c_name:
                    # print("new_name:", new_name, " key:", key, data[key].shape)
                    word_matrix.append(data[key])
                    count += 1
                    cnames.append(c_name)
                    break
    print("count:", count)
    # word_matrix = torch.from_numpy(np.array(word_matrix))
    word_matrix = torch.tensor(word_matrix, dtype=torch.float)
    print("word_matrix.shape:", word_matrix.shape)
    # print("cnames:", len(cnames),cnames)
    return cnames, word_matrix


def get_relation_adj(allclasses_names, dataset_name):
    n_classes = len(allclasses_names)
    adj_matrix = np.zeros((n_classes, n_classes))
    classname2idx = dict()
    idx2classname = dict()
    for i in range(n_classes):
        classname = allclasses_names[i]
        classname2idx[classname] = i
        idx2classname[i] = classname

    relation_search_request = 'http://api.conceptnet.io/related/c/en/'
    for row in range(n_classes):
        for col in range(row, n_classes):  # Symmetric matrix
            class1 = allclasses_names[row]
            class2 = allclasses_names[col]
            request_url = relation_search_request + class1 + '?filter=/c/en/' + class2
            flag = True
            while flag:
                try:
                    obj = requests.get(request_url).json()
                except:
                    print(
                        'There are some errors in the retrieving process! class1=%s, class2=%s' % (class1, class2))
                else:
                    flag = False

            if 'related' in obj:
                if len(obj['related']):
                    adj_matrix[row, col] += obj['related'][0]['weight']
                    print('class1=%s, class2=%s, weight=%f' % (class1, class2, obj['related'][0]['weight']))

    for row in range(1, n_classes):
        for col in range(0, row - 1):
            adj_matrix[row][col] = adj_matrix[col][row]
    print('adj_matrix', adj_matrix)
    check_adj(adj_matrix, idx2classname, doprint=False)
    save_dir = os.path.join(BASE_DIR, dataset_name.lower())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    adj_filename = dataset_name + '_class_adj_' + 'byConceptNet5.5_.pkl'
    adj_filepath = os.path.join(save_dir, adj_filename)
    pklSave(adj_filepath, adj_matrix)
    print('save {} done!'.format(adj_filename))


def pklSave(file_name, obj):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def check_adj(adj_matrix, idx2classname, doprint=True):
    '''
        check the adj matrix for 0 items (where the weight of edge is 0)
    '''
    zero_indexes = np.where(adj_matrix == 0.)
    row_index, col_index = zero_indexes
    assert len(row_index) == len(col_index)
    n_zeros_directed = len(row_index)
    print('There are {} zero edges in adj matrix.'.format(n_zeros_directed))
    zero_edges = []
    for row, col in zip(row_index, col_index):
        _edge = (row, col)
        edge_ = (col, row)
        if _edge in zero_edges or edge_ in zero_edges:
            continue
        else:
            zero_edges.append(_edge)
    print('There are {} undirected zero edges in adj matrix:'.format(len(zero_edges)))

    if doprint:
        for edge in zero_edges:
            row, col = edge
            class1 = idx2classname[row]
            class2 = idx2classname[col]
            print('{} <--> {}'.format(class1, class2))


if __name__ == '__main__':
    tuberlin_word_path = os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "word2vec-google-news.npy")
    tuberlin_hpath = os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "hieremb-path.npy")

    sketchy_word_path = os.path.join(BASE_DIR, "..", "data", "Sketchy", "word2vec-google-news.npy")
    sketchy_hpath = os.path.join(BASE_DIR, "..", "data", "Sketchy", "hieremb-path.npy")

    tuberlin_name_id_txt = r"E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\Tuberlin\zeroshot\cname_cid.txt"
    sketchy_name_id_txt = r"E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\Sketchy\zeroshot1\cname_cid.txt"

    # tuberlin_names, tuberlin_word = get_word_matrix(tuberlin_word_path, tuberlin_name_id_txt)
    # print("tuberlin_names:", tuberlin_names)
    # get_relation_adj(tuberlin_names, "TUBerlin")

    sketchy_names, sketchy_word = get_word_matrix(sketchy_word_path, sketchy_name_id_txt)
    print("sketchy_names:", sketchy_names)
    get_relation_adj(sketchy_names, "Sketchy")

