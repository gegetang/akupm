import argparse
import numpy as np
from data_loader import load_data
from train import train
import os
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(555)
class Akupm(object):
    def __init__(self, args, n_entity, n_relation):
        self.n_entity = n_entity  
        self.n_relation = n_relation
        self.dim = args.dim  
        self.n_hop = args.n_hop  
        self.kge_weight = args.kge_weight  
        self.l2_weight = args.l2_weight  
        self.lr = args.lr 
        self.n_memory = args.n_memory   
        self.item_update_mode = args.item_update_mode  
        self.using_all_hops = args.using_all_hops    

        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _build_inputs(self):
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="akupm_items")
        self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="akupm_labels")
        self.memories_h = []
        self.memories_r = []
        self.memories_t = []

        for hop in range(self.n_hop):
            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="akupm_memories_h_" + str(hop)))
            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="akupm_memories_r_" + str(hop)))
            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="akupm_memories_t_" + str(hop)))

    def _build_embeddings(self):
        self.entity_emb_matrix = tf.get_variable(name="akupm_entity_emb_matrix", dtype=tf.float64,
                                                 shape=[self.n_entity, self.dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        self.relation_emb_matrix = tf.get_variable(name="akupm_relation_emb_matrix", dtype=tf.float64,
                                                   shape=[self.n_relation, self.dim, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())

    def _build_model(self):
        self.transform_matrix = tf.get_variable(name="akupm_transform_matrix", shape=[self.dim, self.dim],
                                                dtype=tf.float64,
                                                initializer=tf.contrib.layers.xavier_initializer())

        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r[i]))
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        o_list = self._key_addressing()

        self.scores = tf.squeeze(self.predict(self.item_embeddings, o_list))
        self.scores_normalized = tf.sigmoid(self.scores)

    def _key_addressing(self):
        o_list = []
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)
            v = tf.expand_dims(self.item_embeddings, axis=2)
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)
            probs_normalized = tf.nn.softmax(probs)
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)
            a = self.t_emb_list[hop] * probs_expanded
            aa = self.multihead_attention(a, a)
            o = tf.layers.dense(tf.transpose(aa, [0, 2, 1]), 1, activation=tf.nn.relu)
            o = tf.reduce_sum(o, axis=2)
            self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
            o_list.append(o)
        return o_list

    def multihead_attention(self, queries, keys, num_units=None, num_heads=2, causality=False, dropout_rate=0,
                            is_training=True):
        if num_units is None: 
            num_units = queries.get_shape().as_list()[-1]

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) 
        K = tf.layers.dense(queries, num_units, activation=tf.nn.relu) 
        V = tf.layers.dense(queries, num_units, activation=tf.nn.relu) 

        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0) 
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0) 

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) 

        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  
        key_masks = tf.tile(key_masks, [num_heads, 1]) 
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) 

        paddings = tf.ones_like(outputs) * (-2 ** self.n_memory + 1)  
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  

        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() 
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) 

            paddings = tf.ones_like(masks) * (-2 ** self.n_memory + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs)  

        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) 
        query_masks = tf.tile(query_masks, [num_heads, 1]) 
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) 
        outputs *= query_masks 

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries
        outputs = self.layer_normalize(outputs)

        return outputs

    def layer_normalize(self, inputs, epsilon=1e-8, scope="ln", reuse=None):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = tf.cast(gamma, tf.float64) * normalized + tf.cast(beta, tf.float64)

        return outputs

    def update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = tf.matmul(o, self.transform_matrix)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y = tf.concat([y, o_list[i]], axis=1)

        y = tf.layers.dense(y, self.dim, activation=tf.nn.relu)
        scores = tf.reduce_sum(item_embeddings * y, axis=1)
        return scores

    def _build_loss(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        self.kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        self.kge_loss = -self.kge_weight * self.kge_loss

        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
            if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss

    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)
    
    def eval(self, sess, feed_dict, PR=False, k=100):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        if PR:
            pre = scores.argsort()[-k:][::-1]
            lab = np.ones(len(labels)) * 10
            for i in range(len(pre)):
                lab[pre[i]] = 1
            pre = np.sum(np.equal(labels, lab)) / k   
            rec = np.sum(np.equal(labels, lab)) / np.sum(np.equal(labels, np.ones(len(labels))))
            F1 = (2 * pre * rec) / (pre + rec)
            return auc, acc, pre, rec, F1
        else:
            return auc, acc
