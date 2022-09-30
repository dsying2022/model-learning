import pandas as pd
import numpy as np
import argparse
import random
from model import KGCN
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,f1_score

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

# prepare arguments (hyperparameters)
parser = argparse.ArgumentParser()
'''
#movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
'''
#music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='concat', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=150, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', nargs='*', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')

args = parser.parse_args()

# build dataset and knowledge graph
data_loader = DataLoader(args.dataset)
kg = data_loader.load_kg()
df_dataset = data_loader.load_dataset()
print("df_dataset:",df_dataset)

# Dataset class
class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label

#train:test:val=6:2:2
x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=0.2, shuffle=False, random_state=999)
#val_size = 0.25 x 0.8 = 0.2,train_size = 0.75 x 0.8 = 0.6
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle=False, random_state=999) 

train_dataset = KGCNDataset(x_train)
test_dataset = KGCNDataset(x_test)
val_dataset = KGCNDataset(x_val)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

# prepare network, loss function, optimizer
num_user, num_entity, num_relation = data_loader.get_num()
user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)#优化可训练参数
print('device: ', device)

# train
train_loss_list = []
test_loss_list = []
val_loss_list = []
auc_score_list = []


for epoch in range(args.n_epochs):
    running_loss = 0.0
    for i, (user_ids, item_ids, labels) in enumerate(train_loader):
        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)#tensor 1*batch_size
        optimizer.zero_grad()#把梯度置零
        outputs = net(user_ids, item_ids)#tensor 1*batch_size
        loss = criterion(outputs, labels)#tensor(loss)
        loss.backward()      
        optimizer.step()
        running_loss += loss.item()#loss的值
    
    train_loss_list.append(running_loss / len(train_loader))
        
    # evaluate per every epoch
    with torch.no_grad():
        test_loss = 0
        val_loss = 0
        test_roc = 0
        val_roc = 0
    
        for user_ids, item_ids, labels in test_loader:
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)#tensor(1 x batchsize)
            outputs = net(user_ids, item_ids)#predict scores,tensor(1 x [batchsize])
            test_loss += criterion(outputs, labels).item()
            test_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())

        auc_score_list.append(test_roc / len(test_loader))
       
        for user_ids, item_ids, labels in val_loader:
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)#tensor(1 x batchsize)
            outputs = net(user_ids, item_ids)#predict scores,tensor(1 x [batchsize])
            val_loss += criterion(outputs, labels).item()
            val_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            
        auc_score_list.append(val_roc / len(val_loader))
        test_loss_list.append(test_loss / len(test_loader))
        val_loss_list.append(val_loss / len(val_loader))
    print("epoch:",epoch)

print("auc_score_list",auc_score_list)
print("{}-dimension of embedding:{}".format(args.dataset,args.dim))
print("auc-score:",np.mean(auc_score_list)) 


