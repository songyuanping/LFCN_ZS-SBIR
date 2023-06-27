import pickle
import numpy as np
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# path = r"data/coco/coco_adj.pkl"
# f = open(path, "rb")
# data = pickle.load(f, encoding="latin1")
# print(data["nums"])
# print(data["adj"].shape)
# print(np.sum(data["adj"], axis=1))
# print(np.sum(data["adj"], axis=0) == np.sum(data["adj"], axis=1))
# for i in range(data["adj"].shape[0]):
#     print("i:",i,data["adj"][i][i])
#     # adj_matrix[i][i]=0

# path = r"data/coco/coco_glove_word2vec.pkl"
# f = open(path, "rb")
# data = pickle.load(f, encoding="latin1")
# print("data.shape:",data.shape)
# print(data["adj"].shape)

# import torch
# path = r"E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\TUBerlin\zeroshot\cid_mask.pickle"
# # path = r"E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\Sketchy\zeroshot1\cid_mask.pickle"
# f = open(path, "rb")
# data = pickle.load(f, encoding="latin1")
# print(data.keys())
# mask_matrix=[]
# for key in data.keys():
#     # print("key:",key,"sum(data[key]):",sum(data[key]),data[key].shape)
#     mask_matrix.append(torch.from_numpy(data[key]))
# mask_matrix=torch.stack(mask_matrix)
# adj_matrix=torch.mm(mask_matrix*10,10*mask_matrix.T).int()
# print("adj_matrix.shape:",torch.sum(adj_matrix,dim=1)==torch.sum(adj_matrix,dim=0))
# print("adj_matrix.shape:",torch.sum(adj_matrix,dim=0))
# print("mask_matrix.shape:",mask_matrix.shape,adj_matrix.shape)
# for i in range(adj_matrix.shape[0]):
#     # print("i:",i,adj_matrix[i][i])
#     adj_matrix[i][i]=0
# print("adj_matrix.shape:",torch.sum(adj_matrix,dim=1)==torch.sum(adj_matrix,dim=0))
# print("adj_matrix.shape:",torch.sum(adj_matrix,dim=1))
# # print(np.array(adj_matrix.cpu().numpy()))
# np.save(os.path.join(BASE_DIR,"..","data","TU-Berlin","adj.npy"),{"mat":adj_matrix.cpu().numpy()})
# # np.save("data/Sketchy/adj.npy",adj_matrix.cpu().numpy())

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

    # name_id_txt = r"/home/syp_pyCharm/Datasets/dataset/TUBerlin/zeroshot/cname_cid.txt"
    # name_id_txt = r"E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\Sketchy\zeroshot1\cname_cid.txt"
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
        else:
            for key in data.keys():
                new_name = str(key).replace("_", "").replace("(", "").replace(")", "")
                # print("c_name:", c_name, "new_name:", new_name)
                if new_name == c_name:
                    # print("new_name:", new_name, " key:", key, data[key].shape)
                    word_matrix.append(data[key])
                    count += 1
                    break
    print("count:", count)
    # word_matrix = torch.from_numpy(np.array(word_matrix))
    word_matrix = torch.tensor(word_matrix, dtype=torch.float)
    print("word_matrix.shape:", word_matrix.shape)
    return word_matrix


if __name__ == '__main__':
    tuberlin_word_path = os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "word2vec-google-news.npy")
    tuberlin_hpath = os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "hieremb-path.npy")

    sketchy_word_path = os.path.join(BASE_DIR, "..", "data", "Sketchy", "word2vec-google-news.npy")
    sketchy_hpath = os.path.join(BASE_DIR, "..", "data", "Sketchy", "hieremb-path.npy")

    tuberlin_name_id_txt = r"E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\Tuberlin\zeroshot\cname_cid.txt"
    sketchy_name_id_txt = r"E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\Sketchy\zeroshot1\cname_cid.txt"

    tuberlin_word = get_word_matrix(tuberlin_word_path, tuberlin_name_id_txt)
    tuberlin_h = get_word_matrix(tuberlin_hpath, tuberlin_name_id_txt)
    print("tuberlin_word.shape:", tuberlin_word.shape, "tuberlin_h.shape:", tuberlin_h.shape)
    # tuberlin_feature = torch.cat([tuberlin_h, tuberlin_word], dim=-1)

    simi = torch.cosine_similarity(tuberlin_word.unsqueeze(0), tuberlin_word.unsqueeze(1), dim=-1)- torch.eye(tuberlin_word.shape[0])
    # simi = torch.cosine_similarity(tuberlin_h.unsqueeze(0), tuberlin_h.unsqueeze(1), dim=-1) - torch.eye(tuberlin_h.shape[0])
    print("simi.shape:", simi.shape, simi)
    with open(os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "word2vec_cosine_adj.pkl_emb"), 'wb') as f:
        pickle.dump(simi, f)
    with open(os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "word2vec_cosine_adj.pkl_emb"), 'rb') as f:
        tuberlin_adj_matrix = pickle.load(f)
    print("tuberlin_adj_matrix.shape:", tuberlin_adj_matrix.shape)  # sketchy_feature.shape: torch.Size([100, 654])
    print("tuberlin_adj_matrix:", tuberlin_adj_matrix)  # sketchy_feature.shape: torch.Size([100, 654])
    # print("tuberlin_feature.shape:", torch.max(tuberlin_feature@tuberlin_feature.T))

    sketchy_word = get_word_matrix(sketchy_word_path, sketchy_name_id_txt)
    sketchy_h = get_word_matrix(sketchy_hpath, sketchy_name_id_txt)
    print("sketchy_word.shape:", sketchy_word.shape, "sketchy_h.shape:", sketchy_h.shape)
    # sketchy_feature = torch.cat([sketchy_h, sketchy_word], dim=-1)
    simi=torch.cosine_similarity(sketchy_word.unsqueeze(0), sketchy_word.unsqueeze(1), dim=-1)- torch.eye(sketchy_word.shape[0])
    # simi = 1.0 - torch.cosine_similarity(sketchy_h.unsqueeze(0), sketchy_h.unsqueeze(1), dim=-1)
    print("simi.shape:", simi.shape, simi)
    with open(os.path.join(BASE_DIR, "..", "data", "Sketchy", "word2vec_cosine_adj.pkl_emb"), 'wb') as f:
        pickle.dump(simi, f)
    with open(os.path.join(BASE_DIR, "..", "data", "Sketchy", "word2vec_cosine_adj.pkl_emb"), 'rb') as f:
        tuberlin_adj_matrix = pickle.load(f)
    print("tuberlin_adj_matrix.shape:", tuberlin_adj_matrix.shape)  # sketchy_feature.shape: torch.Size([100, 654])
    print("tuberlin_adj_matrix:", tuberlin_adj_matrix)  # sketchy_feature.shape: torch.Size([100, 654])

    # tuberlin_word_path = os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "glove-wiki-gigaword.npy")
    # tuberlin_word_path = os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "fasttext-wiki-news-subwords.npy")
    # tuberlin_hpath = os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "hieremb-path.npy")
    #
    # # sketchy_word_path = os.path.join(BASE_DIR, "..", "data", "Sketchy", "glove-wiki-gigaword.npy")
    # sketchy_word_path = os.path.join(BASE_DIR, "..", "data", "Sketchy", "fasttext-wiki-news-subwords.npy")
    # sketchy_hpath = os.path.join(BASE_DIR, "..", "data", "Sketchy", "hieremb-path.npy")
    #
    # tuberlin_name_id_txt = r"E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\Tuberlin\zeroshot\cname_cid.txt"
    # sketchy_name_id_txt = r"E:\pyCharm\sourceFiles\Datasets\Flickr25K(TUBerlin-Extended)\dataset\Sketchy\zeroshot1\cname_cid.txt"
    #
    # tuberlin_word = get_word_matrix(tuberlin_word_path, tuberlin_name_id_txt)
    # tuberlin_h = get_word_matrix(tuberlin_hpath, tuberlin_name_id_txt)
    # print("tuberlin_word.shape:", tuberlin_word.shape, "tuberlin_h.shape:", tuberlin_h.shape)
    # tuberlin_h = torch.cat([tuberlin_h, tuberlin_word], dim=-1)
    #
    # # simi = torch.cosine_similarity(tuberlin_word.unsqueeze(0), tuberlin_word.unsqueeze(1), dim=-1) - torch.eye(tuberlin_word.shape[0])
    # simi= torch.cosine_similarity(tuberlin_h.unsqueeze(0), tuberlin_h.unsqueeze(1), dim=-1)- torch.eye(tuberlin_h.shape[0])
    # print("simi.shape:", simi.shape, simi)
    # with open(os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "hieremb-path_cosine_adj.pkl_emb"), 'wb') as f:
    #     pickle.dump(simi, f)
    # with open(os.path.join(BASE_DIR, "..", "data", "TU-Berlin", "hieremb-path_cosine_adj.pkl_emb"), 'rb') as f:
    #     tuberlin_adj_matrix = pickle.load(f)
    # print("tuberlin_adj_matrix.shape:", tuberlin_adj_matrix.shape)  # sketchy_feature.shape: torch.Size([100, 654])
    # print("tuberlin_adj_matrix:", tuberlin_adj_matrix)  # sketchy_feature.shape: torch.Size([100, 654])
    #
    # sketchy_word = get_word_matrix(sketchy_word_path, sketchy_name_id_txt)
    # sketchy_h = get_word_matrix(sketchy_hpath, sketchy_name_id_txt)
    # print("sketchy_word.shape:", sketchy_word.shape, "sketchy_h.shape:", sketchy_h.shape)
    # sketchy_h = torch.cat([sketchy_h, sketchy_word], dim=-1)
    # # simi = torch.cosine_similarity(sketchy_word.unsqueeze(0), sketchy_word.unsqueeze(1), dim=-1)- torch.eye(sketchy_word.shape[0])
    # simi_distance = torch.cosine_similarity(sketchy_h.unsqueeze(0), sketchy_h.unsqueeze(1), dim=-1)- torch.eye(sketchy_h.shape[0])
    # print("simi.shape:", simi.shape, simi)
    # with open(os.path.join(BASE_DIR, "..", "data", "Sketchy", "hieremb-path_cosine_adj.pkl_emb"), 'wb') as f:
    #     pickle.dump(simi, f)
    # with open(os.path.join(BASE_DIR, "..", "data", "Sketchy", "hieremb-path_cosine_adj.pkl_emb"), 'rb') as f:
    #     tuberlin_adj_matrix = pickle.load(f)
    # print("tuberlin_adj_matrix.shape:", tuberlin_adj_matrix.shape)  # sketchy_feature.shape: torch.Size([100, 654])
    # print("tuberlin_adj_matrix:", tuberlin_adj_matrix)  # sketchy_feature.shape: torch.Size([100, 654])
