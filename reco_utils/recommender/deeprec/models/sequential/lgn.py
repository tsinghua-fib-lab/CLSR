# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from time import time
import socket
import os
import _pickle as cPickle
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from reco_utils.recommender.deeprec.deeprec_utils import load_dict

__all__ = ["LGNModel"]


def init_gcn_weight(hidden_dim, n_layers):
    all_weights = dict()
    initializer = tf.random_normal_initializer(stddev=0.01) #tf.contrib.layers.xavier_initializer()
    weight_size_list = [hidden_dim]*(n_layers +1) 
    for k in range(n_layers):
        all_weights['W_gc_%d' %k] = tf.Variable(
            initializer([weight_size_list[k], weight_size_list[k+1]]), name='W_gc_%d' % k)
        all_weights['b_gc_%d' %k] = tf.Variable(
            initializer([weight_size_list[k+1]]), name='b_gc_%d' % k)
    return all_weights


class LGNModel(SequentialBaseModel):

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization of variables for caser

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
        """
        self.hparams = hparams
        dataset_path = os.path.dirname(hparams.user_vocab)
        self.train_file = os.path.join(dataset_path, 'train_data')
        self.path = dataset_path
        super(LGNModel, self).__init__(hparams, iterator_creator, seed=None)


    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        super(LGNModel, self)._build_embedding()

        ## concat id and cate embedding to form node embedding
        item2cate = self.get_item2cate()
        item2cate[0] = 0 # for all unseen item and cate
        for i in range(self.item_vocab_length):
            if i not in item2cate:
                item2cate[i] = 0 # for unseen item in trainset
        sorted_cate = [i[1] for i in sorted(item2cate.items(), reverse=False, key=lambda x:x[0])]
        new_cate_embedding = tf.nn.embedding_lookup(self.cate_lookup, sorted_cate)
        self.item_lookup = tf.concat([self.item_lookup, new_cate_embedding], axis=1)

        self._gcn()
        print('gcn done')

    def _gcn(self):
        self.node_dropout_flag = False
        self.n_fold = 100
        self.n_layers = 2
        self.n_users = self.user_vocab_length
        self.n_items = self.item_vocab_length
        plain_adj, norm_adj, mean_adj, pre_adj = self.get_adj_mat()
        self.print_statistics(norm_adj, 'normed adj matrix')
        self.weights = init_gcn_weight(self.hparams.item_embedding_dim + self.hparams.cate_embedding_dim, self.n_layers)
        self.user_lookup, self.item_lookup = self._create_lightgcn_embed_ui(norm_adj)
        #self.item_lookup = self._create_lightgcn_embed_ii(norm_adj)
        #return self.user_lookup, self.item_lookup

    def _build_seq_graph(self):
        """The main function to create din model.
        
        Returns:
            obj:the output of din section.
        """
        with tf.name_scope('lgn'):
            model_output = tf.reduce_sum(tf.multiply(self.user_embedding, self.item_embedding), axis=1)
        tf.summary.histogram("model_output", model_output)
        return model_output

    def _fcn_net(self, model_output, layer_sizes, scope):

        output = model_output
        self.logit = tf.sigmoid(output)
        return output

    def print_statistics(self, X, string):
        print('>'*10 + string + '>'*10 )
        print('Shape', X.shape)
        print('Average interactions', X.sum(1).mean(0).item())
        nonzero_row_indice, nonzero_col_indice = X.nonzero()
        unique_nonzero_row_indice = np.unique(nonzero_row_indice)
        unique_nonzero_col_indice = np.unique(nonzero_col_indice)
        print('Non-zero rows', float(len(unique_nonzero_row_indice))/float(X.shape[0]))
        print('Non-zero columns', float(len(unique_nonzero_col_indice))/float(X.shape[1]))
        print('Matrix density', float(len(nonzero_row_indice))/float((X.shape[0]*X.shape[1])))
        print('True Average interactions', float(len(nonzero_row_indice))/float(X.shape[0]))


    def _create_lightgcn_embed_ui(self, R):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(R)
        else:
            A_fold_hat = self._split_A_hat(R)

        ego_embeddings = tf.concat([self.user_lookup, self.item_lookup], 0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            #  ego_embeddings = side_embeddings
            ego_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])

            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        #  all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keep_dims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


    def _create_lightgcn_embed_ii(self, R):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(R)
        else:
            A_fold_hat = self._split_A_hat(R)

        ego_embeddings = self.item_lookup
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            #  ego_embeddings = side_embeddings
            ego_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])

            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        #  all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keep_dims=False)
        i_g_embeddings = all_embeddings
        return i_g_embeddings


    def get_R(self):

        R_ui = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        R_ii = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        train_file = self.train_file
        self.userdict = load_dict(self.hparams.user_vocab)
        print(len(self.userdict))
        self.itemdict = load_dict(self.hparams.item_vocab)
        print(len(self.itemdict))
        with open(train_file) as f_train:
            last_uid = None
            train_lines = f_train.readlines()
            line_num = len(train_lines)
            line_count = 1
            for l in train_lines:
                if len(l) == 0: break
                l = l.strip('\n')
                units = l.strip().split("\t")
                uid = units[1]
                uid = self.userdict[uid]
                #  uid = self.userdict[units[1]] if units[1] in self.userdict else 0
                target_item = units[2]
                item_hist_list = units[5].split(",")

                if uid != last_uid and last_uid is not None:

                    for i in last_item_hist_list:
                        i = self.itemdict[i]
                        R_ui[int(uid), int(i)] = 1.


                    for i,j in zip(last_item_hist_list[:-1], last_item_hist_list[1:]):
                        i = self.itemdict[i]
                        j = self.itemdict[j]
                        R_ii[int(i), int(j)] = 1.
                        R_ii[int(j), int(i)] = 1.

                    k = self.itemdict[last_target_item]
                    R_ii[int(j), int(k)] = 1.
                    R_ii[int(k), int(j)] = 1.

                if line_count == line_num:

                    for i in item_hist_list:
                        i = self.itemdict[i]
                        #  i = self.itemdict[i] if i in self.itemdict else 0
                        R_ui[int(uid), int(i)] = 1.

                    for i,j in zip(item_hist_list[:-1], item_hist_list[1:]):
                        i = self.itemdict[i]
                        j = self.itemdict[j]
                        #  i = self.itemdict[i] if i in self.itemdict else 0
                        #  j = self.itemdict[j] if i in self.itemdict else 0
                        R_ii[int(i), int(j)] = 1.
                        R_ii[int(j), int(i)] = 1.

                    k = self.itemdict[target_item]
                    R_ii[int(j), int(k)] = 1.
                    R_ii[int(k), int(j)] = 1.

                last_uid = uid
                last_item_hist_list = item_hist_list
                last_target_item = target_item
                line_count += 1

        return R_ui, R_ii

    def creat_item2cate(self):
        train_file = self.train_file
        self.userdict = load_dict(self.hparams.user_vocab)
        print(len(self.userdict))
        self.itemdict = load_dict(self.hparams.item_vocab)
        print(len(self.itemdict))
        self.catedict = load_dict(self.hparams.cate_vocab)
        print(len(self.catedict))
        with open(train_file) as f_train:
            last_uid = None
            train_lines = f_train.readlines()
            line_num = len(train_lines)
            line_count = 1
            i2c = {}
            for l in train_lines:
                if len(l) == 0:
                    break
                l = l.strip('\n')
                units = l.strip().split("\t")
                uid = units[1]
                uid = self.userdict[uid]
                #  uid = self.userdict[units[1]] if units[1] in self.userdict else 0
                target_item = units[2]
                target_cate = units[3]
                item_hist_list = units[5].split(",")
                cate_hist_list = units[6].split(",")

                if uid != last_uid and last_uid is not None:

                    t_i = self.itemdict[last_target_item]
                    t_c = self.catedict[last_target_cate]
                    i2c[t_i] = t_c

                    for i,c in zip(last_item_hist_list, last_cate_hist_list):
                        i = self.itemdict[i]
                        c = self.catedict[c]
                        i2c[i] = c

                if line_count == line_num:

                    t_i = self.itemdict[target_item]
                    t_c = self.catedict[target_cate]
                    i2c[t_i] = t_c

                    for i,c in zip(item_hist_list, cate_hist_list):
                        i = self.itemdict[i]
                        c = self.catedict[c]
                        i2c[i] = c

                last_uid = uid
                last_target_item = target_item
                last_target_cate = target_cate
                last_item_hist_list = item_hist_list
                last_cate_hist_list = cate_hist_list
                line_count += 1

        return i2c

    def get_item2cate(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        try:
            t1 = time()
            item2cate = load_dict(self.path + '/item2cate')
            print('already load item2cate', len(item2cate), time() - t1)

        except Exception:
            item2cate = self.creat_item2cate()
            cPickle.dump(item2cate, open(self.path + '/item2cate', "wb"))

        return item2cate
 
    def get_adj_mat(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        try:
            t1 = time()
            adj_mat_short = sp.load_npz(self.path + '/s_adj_mat_short.npz')
            norm_adj_mat_short = sp.load_npz(self.path + '/s_norm_adj_mat_short.npz')
            mean_adj_mat_short = sp.load_npz(self.path + '/s_mean_adj_mat_short.npz')
            print('already load adj matrix short', adj_mat_short.shape, time() - t1)
            adj_mat_ui = sp.load_npz(self.path + '/s_adj_mat_ui.npz')
            norm_adj_mat_ui = sp.load_npz(self.path + '/s_norm_adj_mat_ui.npz')
            mean_adj_mat_ui = sp.load_npz(self.path + '/s_mean_adj_mat_ui.npz')
            print('already load adj matrix ui', adj_mat_ui.shape, time() - t1)
            adj_mat_ii = sp.load_npz(self.path + '/s_adj_mat_ii.npz')
            norm_adj_mat_ii = sp.load_npz(self.path + '/s_norm_adj_mat_ii.npz')
            mean_adj_mat_ii = sp.load_npz(self.path + '/s_mean_adj_mat_ii.npz')
            print('already load adj matrix ii', adj_mat_ii.shape, time() - t1)

        except Exception:
            self.R_ui, self.R_ii = self.get_R()
            self.print_statistics(self.R_ui, 'ui matrix')
            self.print_statistics(self.R_ui.T, 'iu matrix')
            self.print_statistics(self.R_ii, 'ii matrix')

            adj_mat_short, norm_adj_mat_short, mean_adj_mat_short = self.create_adj_mat_short(self.R_ui, self.R_ii)
            adj_mat_ui, norm_adj_mat_ui, mean_adj_mat_ui = self.create_adj_mat_ui(self.R_ui)
            adj_mat_ii, norm_adj_mat_ii, mean_adj_mat_ii = self.create_adj_mat_ii(self.R_ii)
            sp.save_npz(self.path + '/s_adj_mat_short.npz', adj_mat_short)
            sp.save_npz(self.path + '/s_norm_adj_mat_short.npz', norm_adj_mat_short)
            sp.save_npz(self.path + '/s_mean_adj_mat_short.npz', mean_adj_mat_short)

            sp.save_npz(self.path + '/s_adj_mat_ui.npz', adj_mat_ui)
            sp.save_npz(self.path + '/s_norm_adj_mat_ui.npz', norm_adj_mat_ui)
            sp.save_npz(self.path + '/s_mean_adj_mat_ui.npz', mean_adj_mat_ui)

            sp.save_npz(self.path + '/s_adj_mat_ii.npz', adj_mat_ii)
            sp.save_npz(self.path + '/s_norm_adj_mat_ii.npz', norm_adj_mat_ii)
            sp.save_npz(self.path + '/s_mean_adj_mat_ii.npz', mean_adj_mat_ii)

        try:
            pre_adj_mat_short = sp.load_npz(self.path + '/s_pre_adj_mat_short.npz')
            pre_adj_mat_ui = sp.load_npz(self.path + '/s_pre_adj_mat_ui.npz')
            pre_adj_mat_ii = sp.load_npz(self.path + '/s_pre_adj_mat_ui.npz')
        except Exception:
            adj_mat_short=adj_mat_short
            rowsum = np.array(adj_mat_short.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_short = d_mat_inv.dot(adj_mat_short)
            norm_adj_short = norm_adj_short.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat_short = norm_adj_short.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_short.npz', pre_adj_mat_short)

            adj_mat_ui=adj_mat_ui
            rowsum = np.array(adj_mat_ui.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_ui = d_mat_inv.dot(adj_mat_ui)
            norm_adj_ui = norm_adj_ui.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat_ui = norm_adj_ui.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_ui.npz', pre_adj_mat_ui)

            adj_mat_ii=adj_mat_ii
            rowsum = np.array(adj_mat_ii.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_ii = d_mat_inv.dot(adj_mat_ii)
            norm_adj_ii = norm_adj_ii.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat_ii = norm_adj_ii.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_ii.npz', pre_adj_mat_ii)

        #return adj_mat_ii, norm_adj_mat_ii, mean_adj_mat_ii, pre_adj_mat_ii
        return adj_mat_ui, norm_adj_mat_ui, mean_adj_mat_ui, pre_adj_mat_ui

    def create_adj_mat_short(self, R_base, R_side):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R_base = R_base.tolil()
        R_side = R_side.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
            R_base[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
            R_base[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
            adj_mat[self.n_users+int(self.n_items*i/5.0):self.n_users+int(self.n_items*(i+1.0)/5), self.n_users:] =\
            R_side[int(self.n_items*i/5.0):int(self.n_items*(i+1.0)/5)]
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        self.print_statistics(adj_mat, 'adj matrix')

        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def create_adj_mat_ui(self, R):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        self.print_statistics(adj_mat, 'adj matrix ui')
            
        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        
        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


    def create_adj_mat_ii(self, R):
        t1 = time()
        adj_mat = R.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        self.print_statistics(adj_mat, 'adj matrix ii')
            
        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        
        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
  
    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = X.shape[0] // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = X.shape[0]
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = X.shape[0] // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = X.shape[0]
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            #  A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))
            A_fold_hat.append(self._dropout_sparse(temp, 1-0, n_nonzero_temp))

        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


