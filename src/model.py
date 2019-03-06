
# coding: utf-8

# In[1]:


import numpy as np 
import tensorflow as tf


# In[5]:


class Learner:
    #构建一个计算图表示神经ILP模型，处理图上相关的操作，包含update, predict
    def __init__(self, option):
        self.seed = option.seed
        self.num_step = option.num_step
        self.num_layer = option.num_layer
        self.rnn_state_size = option.rnn_state_size
        
        self.norm = option.norm
        self.thr = option.thr
        self.dropout = option.dropout
        self.learning_rate = option.learning_rate
        self.accuracy = option.accuracy
        self.top_k = option.top_k
        
        self.num_entity = option.num_entity
        self.num_operator = option.num_operator
        
        self.num_query = option.num_query
        self.query_embed_size = option.query_embed_size
        
        np.random.seed(self.seed)
        self._build_graph()
        
    def _random_uniform_unit(self, r, c):
        #初始化一个随机单元，size为（r，c）
        #map(function, iterable, ...)：得到把ite按照fun进行转换的结果
        #map(square, [1,2,3,4,5] => [1, 4, 9, 16, 25]
        #求二范数，即这里每一行的随机值都会归一化
        bound = 6./ np.sqrt(c)
        init_matrix = np.random.uniform(-bound, bound, (r, c))
        init_matrix = np.array(list(map(lambda row: row / np.linalg.norm(row), init_matrix)))
        return init_matrix
    
    def _clip_if_not_None(self, g, v, low, high):
        """ Clip not-None gradients to (low, high). """
        """ Gradient of T is None if T not connected to the objective. """
        if g is not None:
            return (tf.clip_by_value(g, low, high), v)
        else:
            return (g, v)
    
    
    def _build_input(self):
        self.tails = tf.placeholder(tf.int32, [None])
        self.heads = tf.placeholder(tf.int32, [None])
        self.targets = tf.one_hot(indices = self.heads, depth = self.num_entity)
        
        self.queries = tf.placeholder(tf.int32, [None, self.num_step])
        #每一个查询都有一个embedding
        self.query_embedding_params = tf.Variable(self._random_uniform_unit(
            self.num_query + 1,
            self.query_embed_size,
			),dtype = tf.float32)
        rnn_inputs = tf.nn.embedding_lookup(self.query_embedding_params, self.queries)
        #rnn_inputs size[None, self.num_step, self.query_embed_size]
        
        return rnn_inputs
    
    def _build_graph(self):
        rnn_inputs = self._build_input()
        # rnn_inputs: a list of num_step tensors,
        # each tensor of size (batch_size, query_embed_size).
        
        #[None, self.num_step, self.query_embed_size]=>
        #[self.num_step, None, self.query_embed_size]
        #tf.split([a, b, c], d, axis = 1) => [d, a, b/d, c], 因此需要reshape
        self.rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size])
                           for q in tf.split(rnn_inputs,
                                            self.num_step,
                                            axis = 1)]
        
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_state_size, state_is_tuple = True)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layer, state_is_tuple = True)
        
        init_state = self.cell.zero_state(tf.shape(self.tails)[0], tf.float32)
        
        self.rnn_outputs, self.final_state = tf.nn.static_rnn(
            self.cell, self.rnn_inputs, initial_state = init_state)
        #self.rnn_outputs size = [self.num_step, None, self.rnn_state_size]
        
        self.W = tf.Variable(np.random.randn(self.rnn_state_size,self.num_operator), dtype = tf.float32)
        self.b = tf.Variable(np.zeros((1, self.num_operator)), dtype = tf.float32)
        
        # attention_operators: a list of num_step lists,
        # each inner list has num_operator tensors,
        # each tensor of size (batch_size, 1).
        # Each tensor represents the attention over an operator. 
        #Wx+b size = [None, self.num_operator]
        self.attention_operators = [tf.split(
            tf.nn.softmax(tf.matmul(rnn_output, self.W) + self.b),
            self.num_operator,
            axis = 1)
            for rnn_output in self.rnn_outputs]
        #self.attention_operators size = [self.num_step, self.num_operator, None, 1]
        
        # attention_memories: (will be) a list of num_step tensors,
        # each of size (batch_size, t+1),
        # where t is the current step (zero indexed).
        # Each tensor represents the attention over currently populated memory cells.
        self.attention_memories = []
        
        # memories: (will be) a tensor of size (batch_size, t+1, num_entity),
        # where t is the current step (zero indexed)
        # Then tensor represents currently populated memory cells.
        #[None, 1, self.num_entity]
        self.memories = tf.expand_dims(tf.one_hot(indices = self.tails, depth = self.num_entity), 1)
        
        self.database = {r : tf.sparse_placeholder(dtype = tf.float32, name = "database_%d" % r)
                        for r in range(self.num_operator // 2)}
        
        #tf.stack(list(size = [a, b]), axis = 2) => size [a, b, len(list)]
        for t in range(self.num_step):
            self.attention_memories.append(
                tf.nn.softmax(
                    tf.squeeze(
                        tf.matmul(
                            tf.expand_dims(self.rnn_outputs[t], 1), # size = [None, 1, self.rnn_state_size]
                            tf.stack(self.rnn_outputs[0: t + 1], axis = 2)), # size = [None, self.rnn_state_size, t]
                        squeeze_dims = [1]))) 
            #size = [None, t]
            
            memory_read = tf.squeeze(
                tf.matmul(
                    tf.expand_dims(self.attention_memories[t], 1), #size = [None, 1, t]
                    self.memories), #size = [None, t, self.num_entity]
                squeeze_dims = [1])
            #size = [None, num_entity]
            
            if t < self.num_step - 1:
                database_results = []
                memory_read = tf.transpose(memory_read)
                for r in range(self.num_operator // 2):
                    #zip之后为[(r, att_r), (r', att_r')],每次算出来当前实体表示在一个关系及其逆关系的映射后的实体表示
                    #存在database_results里，最后做加和
                    for op_matrix, op_attn in zip(
                        [self.database[r],
                        tf.sparse_transpose(self.database[r])],
                        [self.attention_operators[t][r], 
                        self.attention_operators[t][r + self.num_operator // 2]]):
                        product = tf.sparse_tensor_dense_matmul(op_matrix, memory_read)
                        database_results.append(tf.transpose(product) * op_attn)
                 
                added_database_results = tf.add_n(database_results)
                if self.norm:
                    added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))                
                if self.dropout > 0.:
                    added_database_results = tf.nn.dropout(added_database_results, keep_prob=1.-self.dropout)

                self.memories = tf.concat(
                    [self.memories,
                     tf.expand_dims(added_database_results, 1)],
                    axis=1)
            
            else:
                self.predictions = memory_read
        
        self.final_loss =  -tf.reduce_sum(self.targets * tf.log(
            tf.maximum(self.predictions, self.thr)), 1)
        
        if not self.accuracy:
            #返回一个bool列表
            self.in_top = tf.nn.in_top_k(
                            predictions=self.predictions, 
                            targets=self.heads, 
                            k=self.top_k)
        else: 
            #tf.nn.top_k 返回(最大的数, 索引)
            _, indices = tf.nn.top_k(self.predictions, self.top_k, sorted = False)
            self.in_top = tf.equal(tf.squeeze(indices), self.heads)
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # gvs为tuples (gradient, variable)组成的列表。
        gvs = self.optimizer.compute_gradients(tf.reduce_mean(self.final_loss))
        capped_gvs = list(map(lambda grad_var: self._clip_if_not_None(grad_var[0], grad_var[1], -5., 5.), gvs))
        self.optimizer_step = self.optimizer.apply_gradients(capped_gvs)
        
    def _run_graph(self, sess, qq, hh, tt, mdb, to_fetch):
        feed = {}
        #list[a,b]*n = [a,b,…,a,b]
        feed[self.queries] = [[q] * (self.num_step - 1) + [self.num_query]
                                  for q in qq]
        
        feed[self.heads] = hh 
        feed[self.tails] = tt 
        for r in range(self.num_operator // 2):
            feed[self.database[r]] = tf.SparseTensorValue(*mdb[r]) 
        fetches = to_fetch
        graph_output = sess.run(fetches, feed)
        return graph_output
    
    def update(self, sess, qq, hh, tt, mdb):
        to_fetch = [self.final_loss, self.in_top, self.optimizer_step]
        fetched = self._run_graph(sess, qq, hh, tt, mdb, to_fetch)
        return fetched[0], fetched[1]
        
    def predict(self, sess, qq, hh, tt, mdb):
        to_fetch = [self.final_loss, self.in_top]
        fetched = self._run_graph(sess, qq, hh, tt, mdb, to_fetch)
        return fetched[0], fetched[1]
    
    def get_predictions_given_queries(self, sess, qq, hh, tt, mdb):
        to_fetch = [self.in_top, self.predictions]
        fetched = self._run_graph(sess, qq, hh, tt, mdb, to_fetch)
        return fetched[0], fetched[1]
    
    def get_attentions_given_queries(self, sess, queries):
        qq = queries
        hh = [0] * len(queries)
        tt = [0] * len(queries)
        mdb = {r: ([(0,0)], [0.], (self.num_entity, self.num_entity)) 
                for r in range(self.num_operator // 2)}
        to_fetch = [self.attention_operators, self.attention_memories]
        fetched = self._run_graph(sess, qq, hh, tt, mdb, to_fetch)
        return fetched[0], fetched[1]


# In[4]:

