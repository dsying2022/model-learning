import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random


class DataLoader:
    '''
    Data Loader class which makes dataset for training / knowledge graph dictionary
    '''
    def __init__(self, data):
        self.cfg = {
            'movie': {
                'item2id_path': 'data/movie/item_index2entity_id.txt',
                'kg_path': 'data/movie/kg.txt',
                'rating_path': 'data/movie/ratings.csv',
                'rating_sep': ',',
                'threshold': 4.0
            },
            'music': {
                'item2id_path': 'data/music/item_index2entity_id.txt',
                'kg_path': 'data/music/kg.txt',
                'rating_path': 'data/music/user_artists.dat',
                'rating_sep': '\t',
                'threshold': 0.0
            }
        }
        self.data = data
        '''
        item_index2entity_id.txt:从原始评分文件中的项目索引到KG中的实体ID的映射，item->entity
        '''
        df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item','id'])
        df_kg = pd.read_csv(self.cfg[data]['kg_path'], sep='\t', header=None, names=['head','relation','tail'])
        df_rating = pd.read_csv(self.cfg[data]['rating_path'], sep=self.cfg[data]['rating_sep'], names=['userID', 'itemID', 'rating'], skiprows=1)
        
        # df_rating['itemID'] and df_item2id['item'] both represents old entity ID
        df_rating = df_rating[df_rating['itemID'].isin(df_item2id['item'])]#在rating文件中过滤出item_index2entity_id.txt中映射到KG上的item（电影id/音乐家id），即df_rating.itemID和df_item2id.item是同样一个item但编号不同
        df_rating.reset_index(inplace=True, drop=True)
        
        self.df_item2id = df_item2id
        self.df_kg = df_kg
        self.df_rating = df_rating
        
        self.user_encoder = LabelEncoder()#自定义user编码器
        self.entity_encoder = LabelEncoder()#自定义entity编码器
        self.relation_encoder = LabelEncoder()#自定义relation编码器

        self._encoding()
    #编码函数，编码了df_rating['userID'],df_item2id['id'], df_kg['head'],df_kg['relation'], df_kg['tail']    
    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        self.user_encoder.fit(self.df_rating['userID']) #'userID', 'itemID', 'rating'，把userID映射为0~21173
        '''
        index userid
          0     2
          1     2
          2     2
          3     2
          ……
          21172     2100
        '''
        # df_item2id['id'] and df_kg[['head', 'tail']] represents new entity ID
        new_pd = pd.concat([self.df_item2id['id'], self.df_kg['head'], self.df_kg['tail']])#纵向拼接
        self.entity_encoder.fit(new_pd)#0~9365
        self.relation_encoder.fit(self.df_kg['relation'])
        
        # encode df_kg
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])#3846
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])

    def _build_dataset(self):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # df_rating update -> df_dataset
        df_dataset = pd.DataFrame()
        df_dataset['userID'] = self.user_encoder.transform(self.df_rating['userID'])
        
        # update to new id
        item2id_dict = dict(zip(self.df_item2id['item'], self.df_item2id['id']))
        self.df_rating['itemID'] = self.df_rating['itemID'].apply(lambda x: item2id_dict[x])#df_rating.itemID和df_item2id.item同步item的编号，只编码了df_item2id['id']，此处itemID无序
        df_dataset['itemID'] = self.entity_encoder.transform(self.df_rating['itemID'])
        df_dataset['label'] = self.df_rating['rating'].apply(lambda x: 0 if x < self.cfg[self.data]['threshold'] else 1)#大于阈值label=1，否则为0
        
        # negative sampling
        df_dataset = df_dataset[df_dataset['label']==1]
        # df_dataset requires columns to have new entity ID
        full_item_set = set(range(len(self.entity_encoder.classes_)))# df_item2id['id'] and df_kg[['head', 'tail']]
        user_list = [] #userID of negative samples
        item_list = []
        label_list = []
        for user, group in df_dataset.groupby(['userID']):#user：0,1,2... group：以userID聚合的一组
            item_set = set(group['itemID'])#items which related to user
            negative_set = full_item_set - item_set#items which not related to user
            negative_sampled = random.sample(negative_set, len(item_set))#the number of nagative samples is as same as positive samples
            user_list.extend([user] * len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0] * len(negative_sampled))
        negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})
        df_dataset = pd.concat([df_dataset, negative])
        # 从df_dataset中返回指定数量的随机样本，frac：选取样本数量百分数，frac全部返回
        df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)
        df_dataset.reset_index(inplace=True, drop=True)
        print('Done')
        return df_dataset
        
        
    def _construct_kg(self):
        '''
        Construct knowledge graph
        Knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        '''
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(self.df_kg)):
            head = self.df_kg.iloc[i]['head']#提取第i行的'head'列数据
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg
        
    def load_dataset(self):
        return self._build_dataset()

    def load_kg(self):
        return self._construct_kg()
    
    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)
    
    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))
