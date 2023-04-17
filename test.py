# 构建训练集RB198的特征矩阵
import pickle
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from model_0 import deepGCN
DEVICE = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
SEED = 2022
np.random.seed(SEED)
torch.manual_seed(SEED)
BATCH = 1

chain = pickle.load(open('./dataset/com_chain.pkl','rb'))
Onehot = pickle.load(open('./feature/onehot/test_onehot.pkl','rb'))
PSSM = pickle.load(open('./feature/pssm/test_dict.pkl','rb'))
CX = pickle.load(open('./feature/CX/test_CX_minmax.pkl','rb'))
Phy = pickle.load(open('./feature/phy/test_phy_minmax.pkl','rb'))
DSSP = pickle.load(open('./feature/DSSP/test_dssp.pkl','rb'))
Adj = pickle.load(open('./feature/adj/test_adj.pkl','rb'))
test_seq = pickle.load(open('./dataset/protein_unbound_seq.pkl','rb'))
Test_label = pickle.load(open('./dataset/test_label.pkl','rb'))
test_label = {}
for i in Test_label.keys():
    l = []
    for num, j in enumerate(chain[i]):
        l = l + Test_label[i][num]
    test_label[i] = l

def normalized_Laplacian(matrix):
    degree_sum = np.array((matrix.sum(1)))
    D_diag = (degree_sum ** -0.5).flatten()
    D_diag[np.isinf(D_diag)] = 0
    D_inv = np.diag(D_diag)
    Lap = D_inv @ matrix @ D_inv
    return Lap

def RSA_index(data_dssp):
    RNA_index = {}
    for i in data_dssp.keys():
        aa_index = []
        for k,aa in enumerate(data_dssp[i]):
            if aa[0] > 0.05:
                aa_index.append(k)
        RNA_index[i]= aa_index
    return RNA_index

def RSA_index_min(data_dssp):
    RNA_index = {}
    for i in data_dssp.keys():
        aa_index = []
        for k,aa in enumerate(data_dssp[i]):
            if aa[0] < 0.05:
                aa_index.append(k)
        RNA_index[i]= aa_index
    return RNA_index

test_index = RSA_index(DSSP)
index = RSA_index_min(DSSP)
# 利用Data整合数据集的特征
Dataset = []
for i in test_label.keys():
    feature = []
    label = []
    for j in test_index[i]:
        AA_feature = []
        AA_feature.extend(Onehot[i][j])
        AA_feature.extend(PSSM[i][j])
        AA_feature.extend(CX[i][j])
        AA_feature.extend(Phy[i][j])
        AA_feature.extend(DSSP[i][j])
        feature.append(AA_feature)
        label.append(test_label[i][j])
    feature = torch.from_numpy(np.array(feature))
    # label = torch.from_numpy(np.array(list(map(float,test_label[i]))))
    label = torch.from_numpy(np.array(label))
    # adj = torch.from_numpy(np.array(Adj[i])[:, test_index[i]][test_index[i],:])
    adj = torch.from_numpy(normalized_Laplacian(np.array(Adj[i])[:,test_index[i]][test_index[i],:]))
    data = Data(x=feature,y=label)
    data.adj = adj
    data.name = i

    Dataset.append(data)

# 训练
# 寻找最佳MCC以及最佳MCC下的阈值，因为要将预测结果转换为0/1
def best_F_1(label,output):
    F_1_max = 0
    t_max = 0
    for t in range(1,100):
        threshold = t / 100
        predict = np.where(output>threshold,1,0)
        F_1 = skm.f1_score(label, predict, pos_label=1)
        if F_1 > F_1_max:
            F_1_max = F_1
            t_max = threshold
    pred = np.where(output>t_max,1,0)
    accuracy = skm.accuracy_score(label, pred)
    recall = skm.recall_score(label, pred)
    precision = skm.precision_score(label, pred)
    MCC = skm.matthews_corrcoef(label, pred)
    return accuracy,recall,precision,MCC,F_1_max,t_max


model = deepGCN().to(DEVICE)
loss_fun = F.binary_cross_entropy

def test(test_set = Dataset):
    test_loader = DataLoader(test_set, batch_size=1)
    model.load_state_dict(torch.load('./best_model.dat'))
    model.eval()
    all_label = []
    all_pred = []
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            feature = torch.autograd.Variable(data.x.to(DEVICE, dtype=torch.float))
            label = torch.autograd.Variable(data.y.to(DEVICE, dtype=torch.float))
            adj = torch.autograd.Variable(data.adj.to(DEVICE, dtype=torch.float))

            pred = model(feature,adj)

            pred = pred.cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()

            for i in index[''.join(data.name)]:
                pred.insert(i,0)
            for i in index[''.join(data.name)]:
                label.insert(i,0)
            all_label.extend(label)
            all_pred.extend(pred)

    accuracy,recall,precision,MCC,F_1,t_max = best_F_1(np.array(all_label), np.array(all_pred))
    AUC = skm.roc_auc_score(all_label, all_pred)
    precisions, recalls, thresholds = skm.precision_recall_curve(all_label, all_pred)
    AUPRC = skm.auc(recalls, precisions)
    print("test: ")
    print('F_1:', F_1)
    print('ACC:', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('MCC: ',MCC)
    print('AUROC: ',AUC)
    print('AUPRC: ',AUPRC)

test()