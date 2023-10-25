import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    
    

# class MultiModalAttention(nn.Module):
#     def __init__(self, image_feature_dim, table_feature_dim, attention_dim):
#         super(MultiModalAttention, self).__init__()
#         self.image_attention = nn.Linear(image_feature_dim, attention_dim)
#         self.table_attention = nn.Linear(table_feature_dim, attention_dim)
#         self.attention_combine = nn.Linear(attention_dim, 1)

#     def forward(self, image_feature, table_feature):
#         image_attention = self.image_attention(image_feature)
#         table_attention = self.table_attention(table_feature)
#         attention = F.tanh(image_attention + table_attention)
#         attention_score = F.softmax(self.attention_combine(attention), dim=0)
#         attended_image_feature = attention_score * image_feature
#         attended_table_feature = attention_score * table_feature
#         concat_feature = torch.cat([attended_image_feature, attended_table_feature], dim=1)
        
#         return concat_feature



## dropout추가해보자
# class MultiModalAttention(nn.Module):
#     def __init__(self, image_feature_dim, table_feature_dim, dropout_rate):
#         super(MultiModalAttention, self).__init__()
#         self.image_attention = nn.Linear(image_feature_dim, 1)
#         self.table_attention = nn.Linear(table_feature_dim, table_feature_dim)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, image_feature, table_feature):
#         # image_feature에 대한 attention score 계산
#         image_attention_score = torch.sigmoid(self.image_attention(image_feature))

#         # table_feature의 각 feature에 대한 attention score 계산
#         table_attention_score = torch.softmax(self.table_attention(table_feature), dim=1)

#         # attention score를 각 feature에 적용
#         attended_image_feature = image_attention_score * image_feature
#         attended_table_feature = table_attention_score * table_feature

#         # attended_image_feature와 attended_table_feature를 합침
#         concat_feature = torch.cat([attended_image_feature, attended_table_feature], dim=1)
#         # Dropout 적용
#         concat_feature = self.dropout(concat_feature)

#         return concat_feature, table_attention_score, image_attention_score

# class MultiModalAttention(nn.Module):
#     def __init__(self, image_feature_dim, table_feature_dim):
#         super(MultiModalAttention, self).__init__()
#         self.image_attention = nn.Linear(image_feature_dim, 1)
#         self.table_attention = nn.Linear(table_feature_dim, table_feature_dim)
        

#     def forward(self, image_feature, table_feature):
#         # image_feature에 대한 attention score 계산
#         image_attention_score = torch.sigmoid(self.image_attention(image_feature))

#         # table_feature의 각 feature에 대한 attention score 계산
#         table_attention_score = torch.softmax(self.table_attention(table_feature), dim=1)

#         # attention score를 각 feature에 적용
#         attended_image_feature = image_attention_score * image_feature
#         attended_table_feature = table_attention_score * table_feature

#         # attended_image_feature와 attended_table_feature를 합침
#         concat_feature = torch.cat([attended_image_feature, attended_table_feature], dim=1)
        

#         return concat_feature, table_attention_score, image_attention_score



class ScaledDotProductAttention(nn.Module):
    def __init__(self, image_feature_dim, table_feature_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.total_feature_dim = image_feature_dim + table_feature_dim
        self.query = nn.Linear(self.total_feature_dim, self.total_feature_dim)
        self.key = nn.Linear(self.total_feature_dim, self.total_feature_dim)
        self.value = nn.Linear(self.total_feature_dim, self.total_feature_dim)

    def forward(self, image_feature, table_feature):
        # 이미지와 테이블 데이터를 하나의 벡터로 합침
        concat_feature = torch.cat([image_feature, table_feature], dim=1)

        # Query, Key, Value 계산
        Q = self.query(concat_feature)
        K = self.key(concat_feature)
        V = self.value(concat_feature)

        # Scaled Dot-Product Attention 계산
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.total_feature_dim, dtype=torch.float32))
        attention_score = torch.softmax(attention_score, dim=-1)
        attended_feature = torch.matmul(attention_score, V)

        return attended_feature, attention_score

class MultiModalAttention(nn.Module):
    def __init__(self, image_feature_dim, table_feature_dim):
        super(MultiModalAttention, self).__init__()
        self.total_feature_dim = image_feature_dim + table_feature_dim
        self.attention = nn.Linear(self.total_feature_dim, self.total_feature_dim)

    def forward(self, image_feature, table_feature):
        # 이미지와 테이블 데이터를 하나의 벡터로 합침
        concat_feature = torch.cat([image_feature, table_feature], dim=1)

        # 어텐션 스코어 계산
        attention_score = torch.softmax(self.attention(concat_feature), dim=1)

        # 어텐션 스코어를 피처에 적용
        attended_feature = attention_score * concat_feature

        return attended_feature, attention_score



class MultiHeadAttention(nn.Module):
    def __init__(self, image_feature_dim, table_feature_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.image_attention = nn.ModuleList([nn.Linear(image_feature_dim, 1) for _ in range(num_heads)])
        self.table_attention = nn.ModuleList([nn.Linear(table_feature_dim, table_feature_dim) for _ in range(num_heads)])

    def forward(self, image_feature, table_feature):
        attended_features = []
        for i in range(self.num_heads):
            # image_feature에 대한 attention score 계산
            image_attention_score = torch.sigmoid(self.image_attention[i](image_feature))

            # table_feature의 각 feature에 대한 attention score 계산
            table_attention_score = torch.softmax(self.table_attention[i](table_feature), dim=1)

            # attention score를 각 feature에 적용
            attended_image_feature = image_attention_score * image_feature
            attended_table_feature = table_attention_score * table_feature

            # attended_image_feature와 attended_table_feature를 합침
            concat_feature = torch.cat([attended_image_feature, attended_table_feature], dim=1)
            attended_features.append(concat_feature)

        # 모든 헤드의 결과를 연결
        multi_head_feature = torch.cat(attended_features, dim=1)
        return multi_head_feature


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        tabular_size = 60 ####### 수정 필요!!
        self.size_arg = size_arg
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        # self.classifiers = nn.Linear(size[1] + tabular_size, n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        
        # ##추가 2023.08.10
        # positive_weight = 2.0
        # weights = torch.tensor([1.0, positive_weight], device='cuda')
        # self.instance_loss_fn = nn.CrossEntropyLoss(weight=weights)
        
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)
        
    def set_classifier(self, tabular_size : int) : 
        self.classifiers = nn.Linear(self.size_dict[self.size_arg][1] + tabular_size, self.n_classes)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, tabular, label=None, instance_eval=False, return_features=False, attention_only=False):
        # forward(self, h, tabular, label=None, instance_eval=False, return_features=False, attention_only=False)
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h)
        image_feature_dim = M.size(1)
        table_feature_dim = tabular.size(1)
        attention_dim=128
        drop_out_rate=0.2
        attention_module =  ScaledDotProductAttention(image_feature_dim, table_feature_dim).to(device)
        #attention_module=MultiHeadAttention(image_feature_dim, table_feature_dim,3 ).to(device)
        concat,  attention_score = attention_module(M, tabular)
        #concat = attention_module(M, tabular)
        print(concat.shape)
        print(attention_score)
        print(attention_score)
        
         
        #concat = torch.cat([M, tabular], dim=1)
        
        # classifier first input dim
        logits = self.classifiers(concat)
        # -> self.classifiers 코드를 수정 (기존 A.dim이 아니라, A.dim + tabular.dim으로 수정)
        
        
        # logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
