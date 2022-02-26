# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from reco_utils.recommender.deeprec.deeprec_utils import load_dict

__all__ = ["NCFModel"]


class NCFModel(SequentialBaseModel):

    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        super(NCFModel, self)._build_embedding()
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.user_embedding_dim = hparams.user_embedding_dim

        with tf.variable_scope("embedding", initializer=self.initializer):
            self.user_gmf_lookup = tf.get_variable(
                name="user_gmf_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.user_mlp_lookup = tf.get_variable(
                name="user_mlp_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.item_gmf_lookup = tf.get_variable(
                name="item_gmf_embedding",
                shape=[self.item_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.item_mlp_lookup = tf.get_variable(
                name="item_mlp_embedding",
                shape=[self.item_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        super(NCFModel, self)._lookup_from_embedding()

        self.user_gmf_embedding = tf.nn.embedding_lookup(
            self.user_gmf_lookup, self.iterator.users
        )
        tf.summary.histogram("user_gmf_embedding_output", self.user_gmf_embedding)

        self.user_mlp_embedding = tf.nn.embedding_lookup(
            self.user_mlp_lookup, self.iterator.users
        )
        tf.summary.histogram("user_mlp_embedding_output", self.user_mlp_embedding)

        self.item_gmf_embedding = tf.nn.embedding_lookup(
            self.item_gmf_lookup, self.iterator.items
        )
        tf.summary.histogram("item_gmf_embedding_output", self.item_gmf_embedding)

        self.item_mlp_embedding = tf.nn.embedding_lookup(
            self.item_mlp_lookup, self.iterator.items
        )
        tf.summary.histogram("item_short_embedding_output", self.item_mlp_embedding)

    def _build_seq_graph(self):
        """The main function to create din model.
        
        Returns:
            obj:the output of din section.
        """
        with tf.name_scope('ncf'):
            self.gmf  = self.user_gmf_embedding * self.item_gmf_embedding
            self.mlp = tf.concat([self.user_mlp_embedding, self.item_mlp_embedding], -1)
            for layer_size in self.hparams.ncf_layer_sizes:
                self.mlp = tf.contrib.layers.fully_connected(
                    self.mlp,
                    num_outputs=layer_size,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer(
                        seed=self.seed
                    ),
                )
        model_output = tf.concat([self.gmf, self.mlp], -1)
        tf.summary.histogram("model_output", model_output)
        return model_output

    def _fcn_net(self, model_output, layer_sizes, scope):

        output = tf.contrib.layers.fully_connected(
            model_output,
            num_outputs=1,
            activation_fn=None,
            biases_initializer=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(
                seed=self.seed
            ),
        )
        self.logit = tf.sigmoid(output)
        return output
