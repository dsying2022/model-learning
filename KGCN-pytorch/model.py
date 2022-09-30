import sys
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy
from aggregator import Aggregator

class KGCN(torch.nn.Module):
    def __init__(self, num_user, num_ent, num_rel, kg, args, device):
        super(KGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter #计算实体表示时的迭代次数
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)#聚合
        
        self._gen_adj()
            
        self.usr = torch.nn.Embedding(num_user, args.dim)#Embedding(1872, 16)
        self.ent = torch.nn.Embedding(num_ent, args.dim)#Embedding(9366, 16)
        self.rel = torch.nn.Embedding(num_rel, args.dim)#Embedding(60, 16)
        
    def _gen_adj(self):
        '''
        生成实体和关系的邻接矩阵
        只关心固定数量的样本
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        
        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)#随机抽取固定样本的neighbor，不重复
            else:#随机抽取neighbor次固定样本，有重复，list
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
                #neighbors:[(relation, tail),(relation, tail)]
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])#KG上第e个实体的邻居实体
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])#KG上第e个实体和邻居的关系
    #outputs = net(user_ids, item_ids)   
    def forward(self, u, v):
        '''
        input: u, v are batch sized indices for users and items
        u: [1, batch_size],userID
        v: [1, batch_size],itemID
        '''
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.view((-1, 1))#tensor([ [] x batchsize])
        v = v.view((-1, 1))#tensor([ [] x batchsize])
        
        # [batch_size, dim]
        user_embeddings = self.usr(u).squeeze(dim = 1)
        #邻居实体、关系
        entities, relations = self._get_neighbors(v)
        item_embeddings = self._aggregate(user_embeddings, entities, relations)
        scores = (user_embeddings * item_embeddings).sum(dim = 1)#tensor(1 x [batchsize])        
        
        return torch.sigmoid(scores)
    
    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []
        
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1)).to(self.device)#tensor([batchsize x neighbor_sample])
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            
        return entities, relations
    #聚合，将实体v及其领域表示聚合为单个矢量
    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]
        
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.relu
            
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):#get-receptive-field
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        
        return entity_vectors[0].view((self.batch_size, self.dim))

    