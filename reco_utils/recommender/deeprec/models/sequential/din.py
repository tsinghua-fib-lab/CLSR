# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sli_rec import (
    SLI_RECModel,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.deeprec_utils import load_dict

__all__ = ["DINModel"]


class DINModel(SLI_RECModel):

    def _build_seq_graph(self):
        """The main function to create din model.
        
        Returns:
            obj:the output of din section.
        """
        with tf.name_scope('din'):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            self.mask = self.iterator.mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.hist_embedding_sum = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)
            attention_output = self._attention_fcn(self.target_item_embedding, hist_input)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        model_output = tf.concat([self.target_item_embedding, self.hist_embedding_sum, att_fea], -1)
        tf.summary.histogram("model_output", model_output)
        return model_output
