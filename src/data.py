
# coding: utf-8

# In[1]:


import numpy as np
import os
import copy
from math import ceil
from collections import Counter


# In[1]:


class Data:
    def __init__(self, folder, seed):
        #初始化参数
        np.random.seed(seed)
        self.seed = seed
        self.query_include_reverse = True
        
        #初始化关系和实体文件
        self.relation_file = os.path.join(folder, "relations.txt")
        self.entity_file = os.path.join(folder, "entities.txt")
        
        #构建关系和实体与id的对应关系，统计关系和实体数量
        self.relation_to_number, self.entity_to_number = self._numerical_encode()
        self.number_to_entity = {v: k for k, v in self.entity_to_number.items()}
        self.number_to_relation = {v : k for k, v in self.relation_to_number.items()}
        self.num_relation = len(self.relation_to_number)
        self.num_query = self.num_relation * 2
        self.num_entity = len(self.entity_to_number)
        
        #初始化数据文件路径
        self.test_file = os.path.join(folder, "test.txt")
        self.train_file = os.path.join(folder, "train.txt")
        self.valid_file = os.path.join(folder, "valid.txt")
            
        #根据数据文件读入数据
        self.test, self.num_test = self._parse_triplets(self.test_file)
        self.train, self.num_train = self._parse_triplets(self.train_file)
        
        #构建验证集，如果有则读入，如果没有则从训练集中切割
        if os.path.isfile(self.valid_file):
            self.valid, self.num_valid = self._parse_triplets(self.valid_file)
        else:
            self.valid, self.train = self._split_valid_from_train()
            self.num_valid = len(self.valid)
            self.num_train = len(self.train)
                
        #初始化事实数据文件路径
        if os.path.isfile(os.path.join(folder, "facts.txt")):
            self.fact_file = os.path.join(folder, "facts.txt")
            self.share_db = True
        else:
            self.train_fact_file = os.path.join(folder, "train_facts.txt")
            self.test_fact_file = os.path.join(folder, "test_facts.txt")
            self.share_db = False
        
        #根据事实数据构建稀疏矩阵原始文件
        if self.share_db:
            self.fact, self.num_fact = self._parse_triplets(self.fact_file)
            self.matrix_db = self._db_to_matrix_db(self.fact)
            self.matrix_db_train = self.matrix_db
            self.matrix_db_test = self.matrix_db
            self.matrix_db_valid = self.matrix_db
        else:
            self.train_fact, self.num_train_fact = self._parse_triplets(self.train_fact_file)
            self.test_fact, self.num_test_fact = self._parse_triplets(self.test_fact_file)
            self.matrix_db_train = self._db_to_matrix_db(self.train_fact)
            self.matrix_db_test = self.matrxi_db(self.test_fact)
            self.matrix_db_valid = self.matrix_db(self.train_fact)
        
        self.num_operator = 2 * self.num_relation
            
        #获取询问的规则
        #zip(*arr)将arr按照第一维拆分开，list(zip(*self.train))为self.train中的relation
        self.query_for_rules = list(set(list(zip(*self.train))[0]) | 
                                    set(list(zip(*self.test))[0]) | 
                                    set(list(zip(*self._augment_with_reverse(self.train)))[0]) | 
                                    set(list(zip(*self._augment_with_reverse(self.test)))[0]))
        
        #self.parser["query"]为关系id到关系字符串的映射，parser["operator"]为关系id到关系id->关系字符串的映射
        self.parser = self._create_parser()
    
    #构建parser，parser["query"]为关系id到关系字符串的映射，parser["operator"]为关系id到关系id->关系字符串的映射？？？
    def _create_parser(self):
        assert(self.num_query == 2 * len(self.relation_to_number) == 2 * self.num_relation)
        parser = {"query" : {}, "operator" : {}}
        for key, value in self.relation_to_number.items():
            parser["query"][value] = key
            parser["query"][value + self.num_relation] = "inv_" + key
        for query in range(self.num_relation):
            d = {}
            for k, v in self.number_to_relation.items():
                d[k] = v
                d[k + self.num_relation] = "inv_" + v
            parser["operator"][query] = d
            parser["operator"][query + self.num_relation] = d
        return parser
        
    #构建关系和实体与数字id的对应关系
    def _numerical_encode(self):
        relation_to_number = {}
        with open(self.relation_file) as f:
            for line in f:
                l = line.strip()
                relation_to_number[l] = len(relation_to_number)
                
        entity_to_number = {}
        with open(self.entity_file) as f:
            for line in f:
                l = line.strip()
                entity_to_number[l] = len(entity_to_number)
        
        return relation_to_number, entity_to_number
    
    #将原始数据读入，转化为（关系，头实体，尾实体）的三元组的list
    def _parse_triplets(self, file):
        output = []
        with open(file) as f:
            for line in f:
                l = line.strip().split("\t")
                assert(len(l) == 3)
                output.append((self.relation_to_number[l[1]],
                              self.entity_to_number[l[0]],
                              self.entity_to_number[l[2]]))
        return output, len(output)
    
    #在训练数据集中划分验证集
    def _split_valid_from_train(self):
        valid = []
        new_train = []
        for fact in self.train:
            dice = np.random.uniform()
            if dice < 0.1:
                valid.append(fact)
            else:
                new_train.append(fact)
        np.random.shuffle(new_train)
        return valid, new_train
    
    #将读入后的（关系，头实体，尾实体）的三元组的list转化为适合tf处理的稀疏矩阵的格式
    def _db_to_matrix_db(self, db):
        matrix_db = {r: ([[0,0]], [0.], [self.num_entity, self.num_entity])
                    for r in range(self.num_relation)}
        for i, fact in enumerate(db):
            rel = fact[0]
            head = fact[1]
            tail = fact[2]
            value = 1
            matrix_db[rel][0].append([head, tail])
            matrix_db[rel][1].append(value)
        return matrix_db
    
    #将两个转化后适合tf处理的稀疏矩阵的格式的数据合并
    def _combine_two_mdbs(self, mdbA, mdbB):
        new_mdb = {}
        for key, value in mdbA.items():
            new_mdb[key] = value
        for key, value in mdbB.items():
            try:
                value_A = mdbA[key]
                new_mdb[key] = [value_A[0] + value[0], value_A[1] + value[1], value_A[2]]
            except KeyError:
                new_mdb[key] = value
        return new_mdb
    
    def reset(self, batch_size):
        self.batch_size = batch_size
        self.train_start = 0
        self.valid_start = 0
        self.test_start = 0
        
        self.num_batch_train = self._count_batch(self.train, batch_size)
        self.num_batch_valid = self._count_batch(self.valid, batch_size)
        self.num_batch_test = self._count_batch(self.test, batch_size)
    
    def _count_batch(self, samples, batch_size):
        relations = list(zip(*samples))[0]#取得所有样本的关系列表[r0,r1,…]
        relations_counts = Counter(relations)#返回一个字典，统计一个ite中每个对象出现的次数
        num_batches = [ceil(1. * x / batch_size) for x in relations_counts.values()]
        return int(sum(num_batches))
        
    #####################################################
    #对当前关系稀疏矩阵表示增加其逆关系的稀疏矩阵表示
    def _augment_with_reverse(self, triplets):
        augmented = []
        for triplet in triplets:
            augmented += [triplet, (triplet[0] + self.num_relation, triplet[2], triplet[1])]
        return augmented
    
    def _next_batch(self, start, size, samples):
        #参数：开始，数据量，数据源
        #返回值：下一个batch开始的index，这个batch的数据，这个batch数据的index
        assert(start < size)
        end = min(start + self.batch_size, size)
        
        next_start = end % size
        this_batch = samples[start:end]
        if self.query_include_reverse:
            this_batch = self._augment_with_reverse(this_batch)
        this_batch_id = range(start, end)
        return next_start, this_batch, this_batch_id
    
    #拆分三元组
    def _triplet_to_feed(self, triplets):
        queries, heads, tails = zip(*triplets)
        return queries, heads, tails
    
    def next_train(self):
        self.train_start, this_batch, this_batch_id = self._next_batch(
            self.train_start, self.num_train, self.train)
        
        matrix_db = self.matrix_db_train
        return self._triplet_to_feed(this_batch), matrix_db
    
    def next_valid(self):
        self.valid_start, this_batch, _ = self._next_batch(
            self.valid_start, self.num_valid, self.valid)

        matrix_db = self.matrix_db_valid
        return self._triplet_to_feed(this_batch), matrix_db
    
    def next_test(self):
        self.test_start, this_batch, _ = self._next_batch(
            self.test_start, self.num_test, self.test)
        
        matrix_db = self.matrix_db_test
        return self._triplet_to_feed(this_batch), matrix_db

