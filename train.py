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
print(DEVICE)
SEED = 1209
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1205)
    torch.cuda.set_device(0)


BATCH = 1
EPOCH = 100
chain = pickle.load(open('./dataset/com_chain.pkl','rb'))
Onehot = pickle.load(open('./feature/onehot/train_onehot.pkl','rb'))
PSSM = pickle.load(open('./feature/pssm/train_dict.pkl','rb'))
CX = pickle.load(open('./feature/CX/train_CX_minmax.pkl','rb'))
Phy = pickle.load(open('./feature/phy/train_phy_minmax.pkl','rb'))
DSSP = pickle.load(open('./feature/DSSP/train_dssp.pkl','rb'))
Adj = pickle.load(open('./feature/adj/train_adj.pkl','rb'))
train_seq = pickle.load(open('./dataset/protein_unbound_seq.pkl','rb'))
Train_label = pickle.load(open('./dataset/train_label.pkl','rb'))
train_label = {}
for i in Train_label.keys():
    l = []
    for num, j in enumerate(chain[i]):
        l = l + Train_label[i][num]
    train_label[i] = l

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
train_index = RSA_index(DSSP)

# 利用Data整合数据集的特征
Dataset = []
for i in train_label.keys():
    feature = []
    RSA = []
    label= []
    for j in train_index[i]:
    # for j in range(len(train_seq[i])):
        AA_feature = []

        AA_feature.extend(Onehot[i][j])
        AA_feature.extend(PSSM[i][j])
        AA_feature.extend(CX[i][j])
        AA_feature.extend(Phy[i][j])
        AA_feature.extend(DSSP[i][j])
        feature.append(AA_feature)
        # RSA.append(DSSP[i][j][0])
        label.append(train_label[i][j])

    feature = torch.from_numpy(np.array(feature))
    # label = torch.from_numpy(np.array(list(map(float,train_label[i])))
    # adj = torch.from_numpy(np.array(Adj[i])[:, train_index[i]][train_index[i],:])
    adj = torch.from_numpy(normalized_Laplacian(np.array(Adj[i]))[:,train_index[i]][train_index[i],:])
    label = torch.from_numpy(np.array(label))

    data = Data(x=feature,y=label)
    data.adj = adj
    data.name = i
    # data.RSA = torch.from_numpy(np.array(RSA))

    Dataset.append(data)

# 切分数据集RB495
train_data_len = 176
split_num = int(0.9*train_data_len)
data_index = np.arange(train_data_len)
np.random.shuffle(data_index)
Train_index = data_index[:split_num]
valid_index = data_index[split_num:]
train_sampler = SubsetRandomSampler(Train_index)
valid_sampler = SubsetRandomSampler(valid_index)
train_loader = DataLoader(dataset=Dataset,batch_size=BATCH,sampler=train_sampler)
valid_loader = DataLoader(dataset=Dataset,batch_size=BATCH,sampler=valid_sampler)

# 寻找最佳F_1以及最佳F_1下的阈值，因为要将预测结果转换为0/1
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




model = deepGCN()
optimizer = optim.Adam(model.parameters(),lr=0.001)
loss_fun = F.binary_cross_entropy

def train_epoch(model,train_loder,optimizer,loss_fun):
    model.to(DEVICE)
    model.train()
    loss = 0
    num = 0
    for step,data in enumerate(train_loder):
        feature = torch.autograd.Variable(data.x.to(DEVICE,dtype=torch.float))
        label = torch.autograd.Variable(data.y.to(DEVICE,dtype=torch.float))
        adj = torch.autograd.Variable(data.adj.to(DEVICE, dtype=torch.float))

        optimizer.zero_grad()
        output = model(feature,adj)

        train_loss = loss_fun(output,label).to(DEVICE)
        train_loss.backward()
        optimizer.step()
        loss = loss + train_loss.item()
        num = num + len(label)
    epoch_loss = loss/num
    return epoch_loss

def valid_epoch(model,valid_loader,loss_fun):
    model.eval()
    loss = 0
    num = 0
    predict = []
    valid_label = []
    with torch.no_grad():
        for step,data in enumerate(valid_loader):
            feature = torch.autograd.Variable(data.x.to(DEVICE, dtype=torch.float))
            label = torch.autograd.Variable(data.y.to(DEVICE, dtype=torch.float))
            adj = torch.autograd.Variable(data.adj.to(DEVICE, dtype=torch.float))
            # 根据RSA的值更改预测的标签
            pred = model(feature,adj)
            valid_loss = loss_fun(pred,label)
            pred = pred.cpu().numpy()
            valid_label.extend(label.cpu().numpy())
            predict.extend(pred)

            loss = loss + valid_loss.item()
            num += len(label)
    epoch_loss = loss/num
    accuracy,recall,precision,MCC,F_1_max,t_max = best_F_1(np.array(valid_label),np.array(predict))
    return epoch_loss,accuracy,recall,precision,MCC,F_1_max,t_max


loss_t_list = []
loss_v_list = []
F_1_max_epoch = 0
F_1 = []
MCC_list = []
for epoch in range(EPOCH):
    loss_t = train_epoch(model,train_loader,optimizer,loss_fun)
    epoch_loss,accuracy,recall,precision,MCC,F_1_max,t_max = valid_epoch(model,valid_loader,loss_fun)
    F_1.append(F_1_max)
    MCC_list.append(MCC)
    if F_1_max > F_1_max_epoch:
        torch.save(model.cpu().state_dict(), './best_model.dat')
        F_1_max_epoch = F_1_max
    print('EPOCH:',epoch+1)
    print('train_loss:',loss_t)
    print('valid_loss:',epoch_loss)
    print('accuracy:',accuracy)
    print('recall:',recall)
    print('precision:',precision)
    print('MCC:',MCC)
    print('F_1_max:',F_1_max)
    print('t_max:',t_max)
    loss_t_list.append(loss_t)
    loss_v_list.append(epoch_loss)

plt.plot(loss_t_list, 'r-')
plt.title('loss_t_list')
plt.savefig("./loss_t.png")
plt.close()

plt.plot(loss_v_list, 'r-')
plt.title('loss_v_list')
plt.savefig("./loss_v.png")
plt.close()

plt.plot(F_1, 'r-')
plt.title('F_1_list')
plt.savefig("./F_1.png")
plt.close()

plt.plot(MCC_list, 'r-')
plt.title('MCC_list')
plt.savefig("./MCC.png")
plt.close()

