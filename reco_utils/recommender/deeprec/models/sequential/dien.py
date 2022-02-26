# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sli_rec import (
    SLI_RECModel,
)
from tensorflow.contrib.rnn import GRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    VecAttGRUCell,
)
from reco_utils.recommender.deeprec.deeprec_utils import load_dict

from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn

__all__ = ["DIENModel"]


class DIENModel(SLI_RECModel):

    def _build_seq_graph(self):
        """The main function to create din model.
        
        Returns:
            obj:the output of din section.
        """
        hparams = self.hparams
        with tf.name_scope('dien'):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            self.mask = self.iterator.mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.hist_embedding_sum = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)
            with tf.name_scope('rnn_1'):
                self.mask = self.iterator.mask
                self.sequence_length = tf.reduce_sum(self.mask, 1)
                rnn_outputs, _ = dynamic_rnn(
                    GRUCell(hparams.hidden_size),
                    inputs=hist_input,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="gru1"
                )
                tf.summary.histogram('GRU_outputs', rnn_outputs)        

            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                _, alphas = self._attention_fcn(self.target_item_embedding, rnn_outputs, return_alpha=True)

            with tf.name_scope('rnn_2'):
                _, final_state = dynamic_rnn(
                    VecAttGRUCell(hparams.hidden_size),
                    inputs=rnn_outputs,
                    att_scores = tf.expand_dims(alphas, -1),
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="gru2"
                )
                tf.summary.histogram('GRU2_Final_State', final_state)

        model_output = tf.concat([self.target_item_embedding, final_state, self.hist_embedding_sum, self.target_item_embedding*self.hist_embedding_sum], 1)
        tf.summary.histogram("model_output", model_output)
        return model_output
