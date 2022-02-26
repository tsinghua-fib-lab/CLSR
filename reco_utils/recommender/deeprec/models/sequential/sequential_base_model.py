# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import socket

from reco_utils.recommender.deeprec.models.base_model import BaseModel
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric, cal_weighted_metric, cal_mean_alpha_metric, load_dict

__all__ = ["SequentialBaseModel"]


class SequentialBaseModel(BaseModel):
    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all sequential models, such as loss function, 
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.hparams = hparams

        self.need_sample = hparams.need_sample
        self.train_num_ngs = hparams.train_num_ngs
        if self.train_num_ngs is None:
            raise ValueError(
                "Please confirm the number of negative samples for each positive instance."
            )
        self.min_seq_length = (
            hparams.min_seq_length if "min_seq_length" in hparams else 1
        )
        self.hidden_size = hparams.hidden_size if "hidden_size" in hparams else None
        self.graph = tf.Graph() if not graph else graph

        with self.graph.as_default():
            self.embedding_keeps = tf.placeholder(tf.float32, name="embedding_keeps")
            self.embedding_keep_prob_train = None
            self.embedding_keep_prob_test = None

        super().__init__(hparams, iterator_creator, graph=self.graph, seed=seed)

    @abc.abstractmethod
    def _build_seq_graph(self):
        """Subclass will implement this."""
        pass

    def _build_graph(self):
        """The main function to create sequential models.
        
        Returns:
            obj:the prediction score make by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        self.embedding_keep_prob_train = 1.0 - hparams.embedding_dropout
        self.embedding_keep_prob_test = 1.0

        with tf.variable_scope("sequential") as self.sequential_scope:
            self._build_embedding()
            self._lookup_from_embedding()
            model_output = self._build_seq_graph()
            logit = self._fcn_net(model_output, hparams.layer_sizes, scope="logit_fcn")
            self._add_norm()
            return logit

    def train(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        return super(SequentialBaseModel, self).train(sess, feed_dict)

    def batch_train(self, file_iterator, train_sess):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        step = 0
        epoch_loss = 0
        for batch_data_input in file_iterator:
            if batch_data_input:
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, summary) = step_result
                if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )

        return epoch_loss

    def fit(
        self, train_file, valid_file, valid_num_ngs, eval_metric="group_auc" 
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.
        
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.
            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.

        Returns:
            obj: An instance of self.
        """

        # check bad input.
        if not self.need_sample and self.train_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for training without sampling needed."
            )
        if valid_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for validation."
            )

        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1

        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)

            self.writer = tf.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        train_sess = self.sess
        eval_info = list()

        best_metric, self.best_epoch = 0, 0

        for epoch in range(1, self.hparams.epochs + 1):
            self.hparams.current_epoch = epoch
            file_iterator = self.iterator.load_data_from_file(
                train_file,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            )

            epoch_loss = self.batch_train(file_iterator, train_sess)

            valid_res = self.run_weighted_eval(valid_file, valid_num_ngs)
            print(
                "eval valid at epoch {0}: {1}".format(
                    epoch,
                    ",".join(
                        [
                            "" + str(key) + ":" + str(value)
                            for key, value in valid_res.items()
                        ]
                    ),
                )
            )
            eval_info.append((epoch, valid_res))

            progress = False
            early_stop = self.hparams.EARLY_STOP
            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                progress = True
            else:
                if early_stop > 0 and epoch - self.best_epoch >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))
                    break

            if self.hparams.save_model and self.hparams.MODEL_DIR:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if progress:
                    checkpoint_path = self.saver.save(
                        sess=train_sess,
                        save_path=self.hparams.MODEL_DIR + "epoch_" + str(epoch),
                    )

        if self.hparams.write_tfevents:
            self.writer.close()

        print(eval_info)
        print("best epoch: {0}".format(self.best_epoch))
        return self

    def run_eval(self, filename, num_ngs):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                step_pred, step_labels = self.eval(load_sess, batch_data_input)
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        return res

    def eval(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).eval(sess, feed_dict)

    def run_weighted_eval(self, filename, num_ngs, calc_mean_alpha=False, manual_alpha=False):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        users = []
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
        if calc_mean_alpha:
            alphas = []

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                if not calc_mean_alpha:
                    step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input)
                else:
                    step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess, batch_data_input)
                    alphas.extend(np.reshape(step_alpha, -1))
                users.extend(np.reshape(step_user, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        res_weighted = cal_weighted_metric(users, preds, labels, self.hparams.weighted_metrics)
        res.update(res_weighted)
        if calc_mean_alpha:
            if manual_alpha:
                alphas = alphas[0]
            res_alpha = cal_mean_alpha_metric(alphas, labels)
            res.update(res_alpha)
        return res

    def eval_with_user(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.pred, self.iterator.labels], feed_dict=feed_dict)

    def eval_with_user_and_alpha(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.pred, self.iterator.labels, self.alpha_output], feed_dict=feed_dict)

    def predict(self, infile_name, outfile_name):
        """Make predictions on the given data, and output predicted scores to a file.
        
        Args:
            infile_name (str): Input file name.
            outfile_name (str): Output file name.

        Returns:
            obj: An instance of self.
        """

        load_sess = self.sess
        with tf.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input in self.iterator.load_data_from_file(
                infile_name, batch_num_ngs=0
            ):
                if batch_data_input:
                    step_pred = self.infer(load_sess, batch_data_input)
                    step_pred = np.reshape(step_pred, -1)
                    wt.write("\n".join(map(str, step_pred)))
                    wt.write("\n")
        return self

    def infer(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).infer(sess, feed_dict)

    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.cate_vocab_length = len(load_dict(hparams.cate_vocab))
        self.user_embedding_dim = hparams.user_embedding_dim
        self.item_embedding_dim = hparams.item_embedding_dim
        self.cate_embedding_dim = hparams.cate_embedding_dim

        with tf.variable_scope("embedding", initializer=self.initializer):
            self.user_lookup = tf.get_variable(
                name="user_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.item_lookup = tf.get_variable(
                name="item_embedding",
                shape=[self.item_vocab_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
            self.cate_lookup = tf.get_variable(
                name="cate_embedding",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        self.user_embedding = tf.nn.embedding_lookup(
            self.user_lookup, self.iterator.users
        )
        tf.summary.histogram("user_embedding_output", self.user_embedding)

        self.item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.items
        )
        self.item_history_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_history
        )
        tf.summary.histogram(
            "item_history_embedding_output", self.item_history_embedding
        )

        self.cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.cates
        )
        self.cate_history_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_history
        )
        tf.summary.histogram(
            "cate_history_embedding_output", self.cate_history_embedding
        )

        involved_items = tf.concat(
            [
                tf.reshape(self.iterator.item_history, [-1]),
                tf.reshape(self.iterator.items, [-1]),
            ],
            -1,
        )
        self.involved_items, _ = tf.unique(involved_items)
        involved_item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.involved_items
        )
        self.embed_params.append(involved_item_embedding)

        involved_cates = tf.concat(
            [
                tf.reshape(self.iterator.item_cate_history, [-1]),
                tf.reshape(self.iterator.cates, [-1]),
            ],
            -1,
        )
        self.involved_cates, _ = tf.unique(involved_cates)
        involved_cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.involved_cates
        )
        self.embed_params.append(involved_cate_embedding)

        self.target_item_embedding = tf.concat(
            [self.item_embedding, self.cate_embedding], -1
        )
        tf.summary.histogram("target_item_embedding_output", self.target_item_embedding)

        # dropout after embedding
        self.user_embedding = self._dropout(
            self.user_embedding, keep_prob=self.embedding_keeps
        )
        self.item_history_embedding = self._dropout(
            self.item_history_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_history_embedding = self._dropout(
            self.cate_history_embedding, keep_prob=self.embedding_keeps
        )
        self.target_item_embedding = self._dropout(
            self.target_item_embedding, keep_prob=self.embedding_keeps
        )

    def _add_norm(self):
        """Regularization for embedding variables and other variables."""
        all_variables, embed_variables = (
            tf.trainable_variables(),
            tf.trainable_variables(self.sequential_scope._name + "/embedding"),
        )
        layer_params = list(set(all_variables) - set(embed_variables))
        self.layer_params.extend(layer_params)
