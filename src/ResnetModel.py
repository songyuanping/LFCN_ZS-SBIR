import pretrainedmodels
import torch
import pickle
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from resnet import resnet50_hashing
from senet import cse_resnet50, cse_resnet50_hashing
from gtn_models.gtn import GTLayer
from gtn_models.util import gen_A, gen_emb_A, gen_adj


class ResnetModel(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False):
        super(ResnetModel, self).__init__()

        self.num_classes = num_classes
        self.modelName = arch

        original_model = models.__dict__[arch](pretrained=pretrained)
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.last_block = nn.Sequential(*list(original_model.children())[-3:-1])
        self.linear = nn.Linear(in_features=2048, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False

    def forward(self, x):
        out = self.features(x)
        out = self.last_block(out)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)
        return out


class ResnetModel_KDHashing(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(ResnetModel_KDHashing, self).__init__()
        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            self.original_model = resnet50_hashing(self.hashing_dim)
        else:
            self.original_model = resnet50_hashing(self.hashing_dim, pretrained=False)

        self.ems = ems
        if self.ems:
            print('Error, no ems implementationin AlexnetModel_KDHashing')
            return None
        else:
            self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            print('Error, no freeze_features implementationin AlexnetModel_KDHashing')
            return None

    def forward(self, x):
        out_o = self.original_model.features(x)
        out_o = self.original_model.hashing(out_o)

        out = self.linear(out_o)
        out_kd = self.original_model.logits(out_o)

        #         x_norm = out_o.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        #         normed = out_o / x_norm

        #         out = F.linear(normed, F.normalize(self.linear.weight), None) * x_norm
        #         out_kd = F.linear(normed, F.normalize(self.original_model.last_linear.weight), None) * x_norm

        return out, out_kd


class SEResnetModel(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(SEResnetModel, self).__init__()

        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
        else:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained=None)

        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.last_block = nn.Sequential(*list(original_model.children())[-3:-1])

        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, 2048)
        else:
            self.linear = nn.Linear(in_features=2048, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False

    def forward(self, x):
        out = self.features(x)
        out = self.last_block(out)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)
        return out


class SEResnetModel_KD(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(SEResnetModel_KD, self).__init__()

        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
        else:
            original_model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained=None)

        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.last_block = nn.Sequential(*list(original_model.children())[-3:-1])
        self.original_output = nn.Sequential(*list(original_model.children())[-1:])

        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, 2048)
        else:
            self.linear = nn.Linear(in_features=2048, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False

    def forward(self, x):
        out_o = self.features(x)
        out_o = self.last_block(out_o)
        out_o = out_o.view(out_o.size()[0], -1)

        out = self.linear(out_o)
        out_kd = self.original_output(out_o)

        return out, out_kd


class CSEResnetModel_KD(nn.Module):
    def __init__(self, arch, num_classes, pretrained=True, freeze_features=False, ems=False):
        super(CSEResnetModel_KD, self).__init__()

        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            self.original_model = cse_resnet50()
        else:
            self.original_model = cse_resnet50(pretrained=None)

        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, 2048)
        else:
            self.linear = nn.Linear(in_features=2048, out_features=num_classes)

        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False

    def forward(self, x, y):
        out_o = self.original_model.features(x, y)
        out = nn.AdaptiveAvgPool2d(1)(out_o)
        out = out.view(out.size()[0], -1)
        features = out
        out = self.linear(out)

        out_kd = self.original_model.logits(out_o)
        return out, out_kd, features


from torch.nn import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print("input.shape, input:", input.shape, self.weight.shape)
        support = torch.matmul(input, self.weight)
        # print("input.shape, support:", input.shape, support.shape)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, num_classes=220, feature_dim=2048, gc_in_channel=300, t=0.4, adj_files=None):
        super(GCN, self).__init__()
        self.num_classes = num_classes
        self.gc1 = GraphConvolution(gc_in_channel, 1024)
        self.gc2 = GraphConvolution(1024, feature_dim)
        self.relu = nn.LeakyReLU(0.2)

        A_Tensor = torch.eye(num_classes).type(torch.FloatTensor).unsqueeze(-1)

        for adj_file in adj_files:
            if '_emb' in adj_file:
                _adj = gen_emb_A(adj_file)
            else:
                _adj = gen_A(num_classes, t, adj_file)

            # _adj = torch.from_numpy(_adj).type(torch.FloatTensor)
            _adj = torch.tensor(_adj).type(torch.FloatTensor)
            A_Tensor = torch.cat([A_Tensor, _adj.unsqueeze(-1)], dim=-1)

        # print("A_Tensor.shape:", A_Tensor.shape)
        # A_Tensor.shape: torch.Size([220, 220, 4])
        self.gtn = GTLayer(A_Tensor.shape[-1], 1, first=True)
        self.A = A_Tensor.unsqueeze(0).permute(0, 3, 1, 2)
        # print("self.A.shape:", self.A.shape)
        # self.A.shape: torch.Size([1, 4, 220, 220])

    def forward(self, vis_feature, vec_matrix):
        adj, _ = self.gtn(self.A)
        # print("adj.shape:", adj.shape)
        # adj.shape: torch.Size([1, 220, 220])
        adj = torch.squeeze(adj, 0) + torch.eye(self.num_classes).type(torch.FloatTensor).cuda()
        adj = gen_adj(adj)
        # print("adj.shape:", adj.shape)
        # adj.shape: torch.Size([220, 220])

        sem_x = self.gc1(vec_matrix, adj)
        sem_x = self.relu(sem_x)
        # print("sem_x.shape:", sem_x.shape)
        sem_x = self.gc2(sem_x, adj)
        # print("sem_x.shape:", sem_x.shape)
        sem_x = sem_x.transpose(0, 1)
        out = torch.matmul(vis_feature, sem_x)
        return out


class CSEResnetModel_KDHashing(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, freeze_features=False, ems=False, module='CSE'):
        super(CSEResnetModel_KDHashing, self).__init__()

        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            self.original_model = cse_resnet50_hashing(self.hashing_dim, module=module)
        else:
            self.original_model = cse_resnet50_hashing(self.hashing_dim, pretrained=None, module=module)

        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, hashing_dim)
        else:
            # self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes)
            # 改进
            self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes, bias=False)

        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False

    def forward(self, x, y):
        out_o = self.original_model.features(x, y)

        features_map = out_o
        out_o, features = self.original_model.hashing(out_o)
        hash_code = out_o

        # 原实现
        # out = self.linear(out_o)
        # 替换为
        # sem_x = self.gcn_forward(vec_matrix)
        # print("x.shape:", x.shape, "feature.shape:", feature.shape)
        # x.shape: torch.Size([2048, 80])
        # feature.shape: torch.Size([3, 2048])

        # print("x.shape:", x.shape)
        # x.shape: torch.Size([3, 80])

        out_kd = self.original_model.logits(out_o)

        #         x_norm = out_o.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        #         normed = out_o / x_norm

        #         out = F.linear(normed, F.normalize(self.linear.weight), None) * x_norm
        #         out_kd = F.linear(normed, F.normalize(self.original_model.last_linear.weight), None) * x_norm
        # print("out.shape, out_kd.shape, hash_code.shape, features.shape: ",out.shape, out_kd.shape, hash_code.shape, features.shape)
        return out_kd, hash_code, features, features_map


class DN_CSEResnetModel_KDHashing(nn.Module):
    def __init__(self, arch, hashing_dim, num_classes, pretrained=True, freeze_features=False, ems=False, module='CSE',
                 neighbor_k=3):
        super(DN_CSEResnetModel_KDHashing, self).__init__()

        self.hashing_dim = hashing_dim
        self.num_classes = num_classes
        self.modelName = arch

        if pretrained:
            self.original_model = cse_resnet50_hashing(self.hashing_dim, module=module)
        else:
            self.original_model = cse_resnet50_hashing(self.hashing_dim, pretrained=None, module=module)

        self.ems = ems
        if self.ems:
            self.linear = EMSLayer(num_classes, hashing_dim)
        else:
            # self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes)
            # 改进
            self.linear = nn.Linear(in_features=hashing_dim, out_features=num_classes, bias=False)

        # Freeze the resnet layers
        if freeze_features:
            for ff in self.features:
                for pp in ff.parameters():
                    pp.requires_grad = False

        self.imgtoclass = ImgtoClass_Metric(neighbor_k=neighbor_k)  # 1*num_classes

    def image_to_class(self, input1, input2):

        # extract features of input1--query image
        q = input1

        # extract features of input2--support set
        S = []
        for i in range(len(input2)):
            support_set_sam = input2[i]
            B, C, h, w = support_set_sam.size()  # support_set_sam.shape: torch.Size([5, 64,21,21])
            support_set_sam = support_set_sam.permute(1, 0, 2, 3)  # support_set_sam.shape: torch.Size([64,5, 21,21])
            support_set_sam = support_set_sam.contiguous().view(C, -1)  # support_set_sam.shape: torch.Size([64,2205])
            S.append(support_set_sam)

        x = self.imgtoclass(q, S)  # get Batch*num_classes

        return x

    def forward(self, input_var1, tag_var1, input_var2, tag_var2):
        # print("input_var1.shape:", input_var1.shape,"tag_var1.shape:", tag_var1.shape)
        # print("tag_var2.shape:", tag_var2.shape)
        out_var1 = self.original_model.features(input_var1, tag_var1)
        # print("out_var1.shape:", out_var1.shape)
        out_var2 = []
        for i in range(len(input_var2)):
            # print("input2.shape:", input_var2[i].shape, "tag2.shape:", tag_var2.shape)
            support_set_sam = self.original_model.features(input_var2[i], tag_var2)
            # print("support_set_sam.shape:", support_set_sam.shape)
            out_var2.append(support_set_sam)
        batch_label_target1 = self.image_to_class(out_var1, out_var2)
        # print("out_var2[0].shape:", out_var2[0].shape)
        out_var1, features1 = self.original_model.hashing(out_var1)
        hash_code1 = out_var1
        hash_code2, features2 = [], []
        for i in range(len(out_var2)):
            code_2, feature2 = self.original_model.hashing(out_var2[i])
            hash_code2.append(code_2)
            features2.append(feature2)
        # print("hash_code2[0].shape:", hash_code2[0].shape)
        # print("features2[0].shape:", features2[0].shape)
        out1 = self.linear(out_var1)
        out_kd1 = self.original_model.logits(out_var1)
        out2, out_kd2 = [], []
        for i in range(len(hash_code2)):
            out, kd = self.linear(hash_code2[i]), self.original_model.logits(hash_code2[i])
            out2.append(out)
            out_kd2.append(kd)
        # print("out2[0].shape:", out2[0].shape)
        # print("out_kd2[0].shape:", out_kd2[0].shape)
        return out1, out_kd1, hash_code1, features1, batch_label_target1, out2, out_kd2, hash_code2, features2


class EMSLayer(nn.Module):
    def __init__(self, num_classes, num_dimension):
        super(EMSLayer, self).__init__()
        self.cpars = torch.nn.Parameter(torch.randn(num_classes, num_dimension))
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        out = pairwise_distances(x, self.cpars)
        out = - self.relu(out).sqrt()
        return out


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


# class HashingEncoder(nn.Module):
#     def __init__(self, input_dim, one_dim, two_dim, hash_dim):
#         super(HashingEncoder, self).__init__()
#         self.input_dim = input_dim
#         self.one_dim = one_dim
#         self.two_dim = two_dim
#         self.hash_dim = hash_dim

#         self.en1 = nn.Linear(input_dim, one_dim)
#         self.en2 = nn.Linear(one_dim, two_dim)
#         self.en3 = nn.Linear(two_dim, hash_dim)

#         self.de1 = nn.Linear(hash_dim, two_dim)
#         self.de2 = nn.Linear(two_dim, one_dim)
#         self.de3 = nn.Linear(one_dim, input_dim)

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         e = self.en1(x)
#         e = self.relu(e)
#         e = self.en2(e)
#         e = self.relu(e)
#         e = self.en3(e)

#         # h = self.relu(torch.sign(e))

#         r = self.de1(e)
#         r = self.relu(r)
#         r = self.de2(r)
#         r = self.relu(r)
#         r = self.de3(r)
#         r = self.relu(r)

#         return e, r


class HashingEncoder(nn.Module):
    def __init__(self, input_dim, one_dim, hash_dim):
        super(HashingEncoder, self).__init__()
        self.input_dim = input_dim
        self.one_dim = one_dim
        self.hash_dim = hash_dim

        self.en1 = nn.Linear(input_dim, one_dim)
        self.en2 = nn.Linear(one_dim, hash_dim)
        self.de1 = nn.Linear(hash_dim, one_dim)
        self.de2 = nn.Linear(one_dim, input_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        e = self.en1(x)
        e = self.en2(self.relu(e))
        # e = self.en2(e)
        # e = self.relu(e)
        # e = self.en3(e)

        # h = self.relu(torch.sign(e))

        r = self.de1(e)
        r = self.de2(self.relu(r))
        # r = self.relu(r)
        # r = self.de2(r)
        # r = self.relu(r)
        # r = self.de3(r)
        r = self.relu(r)

        return e, r


class ScatterLoss(nn.Module):
    def __init__(self):
        super(ScatterLoss, self).__init__()

    def forward(self, e, y):
        sample_num = y.shape[0]
        e_norm = e / torch.sqrt(torch.sum(torch.mul(e, e), dim=1, keepdim=True))
        cnter = 0
        loss = 0
        for i1 in range(sample_num - 1):
            e1 = e_norm[i1]
            y1 = y[i1]
            for i2 in range(i1 + 1, sample_num):
                e2 = e_norm[i2]
                y2 = y[i2]
                if y1 != y2:
                    cnter += 1
                    loss += torch.sum(torch.mul(e1, e2))

        return loss / cnter


class QuantizationLoss(nn.Module):
    def __init__(self):
        super(QuantizationLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, e):
        return self.mse(e, torch.sign(e))


# ========================== Define an image-to-class layer ==========================#


class ImgtoClass_Metric(nn.Module):
    def __init__(self, neighbor_k=3):
        super(ImgtoClass_Metric, self).__init__()
        self.neighbor_k = neighbor_k

    # Calculate the k-Nearest Neighbor of each local descriptor
    def cal_cosinesimilarity(self, input1, input2):
        B, C, h, w = input1.size()  # input1.size(): torch.Size([50, 64, 21, 21])
        # print("input1.size():",input1.size())
        Similarity_list = []

        for i in range(B):
            query_sam = input1[i]
            query_sam = query_sam.view(C, -1)
            query_sam = torch.transpose(query_sam, 0, 1)
            query_sam = F.normalize(query_sam, dim=-1)  # query_sam.shape: torch.Size([441, 64])
            # print("query_sam.shape:",query_sam.shape)

            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, len(input2)).cuda()  # inner_sim.shape: torch.Size([1, N_way])

            for j in range(len(input2)):
                support_set_sam = input2[j]  # support_set_sam.shape: torch.Size([64, 2205])
                support_set_sam = F.normalize(support_set_sam, dim=0)  # support_set_sam.shape: torch.Size([64, 2205])
                # print("support_set_sam.shape:", support_set_sam.shape)

                # cosine similarity between a query sample and a support category
                innerproduct_matrix = query_sam @ support_set_sam  # innerproduct_matrix.shape: torch.Size([441, 2205])
                # print("innerproduct_matrix.shape:", innerproduct_matrix.shape)

                # choose the top-k nearest neighbors
                topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k,
                                                    1)  # topk_value: torch.Size([441, 3])
                # print("topk_value:", topk_value.shape)
                inner_sim[0, j] = torch.sum(topk_value)

            Similarity_list.append(inner_sim)

        Similarity_list = torch.cat(Similarity_list, 0)

        return Similarity_list

    def forward(self, x1, x2):

        Similarity_list = self.cal_cosinesimilarity(x1, x2)

        return Similarity_list


# ========================== Define an fusion feature net layer ==========================#


class FusionFeature_Layer(nn.Module):
    def __init__(self):
        super(FusionFeature_Layer, self).__init__()

    def forward(self, query_feature, support_feature):
        """将teacher model得到的特征作为query_feature,student model得到的特征作为support_feature"""
        q, s = query_feature, support_feature
        B, C, h, w = support_feature.size()  # input1.size(): torch.Size([50, 64, 21, 21])
        # print("input1.size():",input1.size())
        query_feature = query_feature.view(B, -1, C)  # [b,49,2048]
        support_feature = support_feature.view(B, C, -1)  # [b,2048,49]
        batch_adj = torch.matmul(query_feature, support_feature)  # [b,49,49]
        query_logits = torch.matmul(torch.softmax(batch_adj, dim=-1), query_feature)  # [b,49,2048]
        attention_mask = torch.sigmoid(batch_adj.sum(dim=-1))  # [b,49]
        query_logits *= attention_mask.unsqueeze(-1)  # [b,49,2048]
        # print(attention_mask.unsqueeze(-1).shape, attention_mask)
        support_feature = support_feature.view(B, -1, C)
        support_logits = support_feature * attention_mask.unsqueeze(-1)  # [b,49,2048]
        query_logits = query_logits.view(B, C, h, w)
        support_logits = support_logits.view(B, C, h, w)
        q = q + query_logits
        s = s + support_logits
        return q, s


if __name__ == '__main__':
    net = FusionFeature_Layer()
    query_logits, support_logits = torch.randn((128, 2048, 7, 7)), torch.randn((128, 2048, 7, 7))
    query_logits, support_logits = net(query_logits, support_logits)
    print(query_logits.shape, support_logits.shape)
    print(query_logits[0, 0], support_logits[0, 0])
