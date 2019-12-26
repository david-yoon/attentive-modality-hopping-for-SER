#-*- coding: utf-8 -*-

"""
what    : Single Encoder Model for text - bidirectional
test     : LTC
data    : IEMOCAP
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper 

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from layers import add_GRU

from project_config import *

class ModelText:
    
    def __init__(self, dic_size,
                 use_glove,
                 batch_size,
                 encoder_size,
                 num_layer, lr,
                 hidden_dim,
                 dr,
                 bi, attn, ltc
                ):
        
        self.dic_size = dic_size
        self.use_glove = use_glove
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.num_layers = num_layer
        self.lr = lr
        self.hidden_dim = hidden_dim

        self.dr_text_in  = dr
        self.dr_text_out = 1.0
        
        self.bi = bi
        self.attn = attn
        
        self.ltc = ltc
        self.dr_text_ltc = LTC_DR_TEXT
        
        self.encoder_inputs = []
        self.encoder_seq_length =[]
        self.y_labels =[]

        self.M = None
        self.b = None
        
        self.y = None
        self.optimizer = None

        self.batch_loss = None
        self.loss = 0
        self.batch_prob = None
        
        if self.use_glove == 1:
            self.embed_dim = 300
        else:
            self.embed_dim = DIM_WORD_EMBEDDING
        
        # for global counter
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    

    def _create_placeholders(self):
        print ('[launch-text] placeholders')
        with tf.name_scope('text_placeholder'):
            
            self.encoder_inputs  = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, self.encoder_size], name="encoder")  # [batch,time_step]
            self.encoder_seq     = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size], name="encoder_seq")   # [batch] - valid word step
            self.y_labels        = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size, N_CATEGORY], name="label")
            
            self.dr_text_in_ph   = tf.compat.v1.placeholder(tf.float32, name="dropout_text_in")
            self.dr_text_out_ph  = tf.compat.v1.placeholder(tf.float32, name="dropout_text_out")
            
            self.dr_text_ltc_ph  = tf.compat.v1.placeholder(tf.float32, name="dropout_ltc")

             # for using pre-trained embedding
            self.embedding_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[self.dic_size, self.embed_dim], name="embedding_placeholder")

    def _create_embedding(self):
        print ('[launch-text] create embedding')
        with tf.name_scope('embed_layer'):
            self.embed_matrix = tf.Variable(tf.random.normal([self.dic_size, self.embed_dim],
                                                            mean=0.0,
                                                            stddev=0.01,
                                                            dtype=tf.float32,                                                             
                                                            seed=None),
                                                            trainable = EMBEDDING_TRAIN,
                                                            name='embed_matrix')
            
            self.embed_en       = tf.nn.embedding_lookup(self.embed_matrix, self.encoder_inputs, name='embed_encoder')
            
            
    def _use_external_embedding(self):
        if self.use_glove == 1:
            print ('[launch-text] use pre-trained embedding')
            self.embedding_init = self.embed_matrix.assign(self.embedding_placeholder)
            
            
    
    def test_cross_entropy_with_logit(self, logits, labels):
        x = logits
        z = labels
        return tf.maximum(x, 0) - x * z + tf.log(1 + tf.exp(-tf.abs(x)))
 


    def _create_gru_model(self):
        print ('[launch-text] create gru cell - bidirectional:', self.bi)

        with tf.name_scope('text_RNN') as scope:
        
            with tf.compat.v1.variable_scope("text_GRU", reuse=False, initializer=tf.orthogonal_initializer()):
                
                # match embedding_dim - rnn_dim to use residual connection            
                if IS_TEXT_RESIDUAL:
                    self.text_residual_matrix = tf.Variable(tf.random.normal([self.embed_dim, self.hidden_dim],
                                                                 mean=0.0,
                                                                 stddev=0.01,
                                                                 dtype=tf.float32,                                                             
                                                                 seed=None),
                                                                 trainable = True,
                                                                 name='text_residual_projection')
                    
                    self.text_residual_bias =  tf.Variable(tf.zeros([self.hidden_dim], dtype=tf.float32), name="text_res_bias")

                    h = tf.matmul( tf.reshape(self.embed_en, [-1, self.embed_dim]), self.text_residual_matrix ) + self.text_residual_bias
                    self.embed_en_match_dim = tf.reshape( h, [self.batch_size, self.encoder_size, self.hidden_dim] )

                else:
                    self.embed_en_match_dim = self.embed_en


                self.outputs, self.output_states = add_GRU(
                                                    inputs= self.embed_en_match_dim,
                                                    inputs_len=self.encoder_seq,
                                                    hidden_dim = self.hidden_dim,
                                                    layers = self.num_layers,
                                                    scope = 'text_encoding_RNN',
                                                    reuse = False,
                                                    dr_input_keep_prob  = self.dr_text_in_ph,
                                                    dr_output_keep_prob = self.dr_text_out_ph,
                                                    is_bidir    = self.bi,
                                                    is_bw_reversed = True,
                                                    is_residual = IS_TEXT_RESIDUAL
                                                    )
                
                self.state_concat  = self.outputs
                self.final_encoder = self.output_states[-1]
                
            if self.bi : self.final_encoder_dimension   = self.hidden_dim * 2
            else       : self.final_encoder_dimension   = self.hidden_dim
        
    
    def _add_attn(self):
        from model_luong_attention import luong_attention
        print ('[launch-text] apply Attention')
        
        with tf.name_scope('text_Attn') as scope:

            # attention memory
            self.attnM = tf.Variable(tf.random.uniform([self.final_encoder_dimension],
                                                       minval= -0.25,
                                                       maxval= 0.25,
                                                       dtype=tf.float32,
                                                       seed=None),
                                                     trainable=True,
                                                     name="attn_memory")

            
            self.attnB =  tf.Variable(tf.zeros([self.final_encoder_dimension], dtype=tf.float32), name="attn_bias")

            # multiply attn memoery as many as batch_size ( use same attn memory )
            self.batch_attnM = tf.ones( [self.batch_size, 1] ) * self.attnM + self.attnB 

            self.final_encoder, self.attn_norm = luong_attention( 
                                                                batch_size = self.batch_size,
                                                                target = self.final_encoder,
                                                                condition = self.batch_attnM ,
                                                                batch_seq = self.encoder_seq,
                                                                max_len = self.encoder_size,
                                                                hidden_dim = self.final_encoder_dimension
                                                                )
        
        
    def _add_LTC_method(self):
        from model_sy_ltc import sy_ltc
        print ('[launch-text] apply LTC method, LTC_DR: ',LTC_DR_TEXT)

        with tf.name_scope('text_LTC') as scope:
            self.final_encoder, self.final_encoder_dimension = sy_ltc( batch_size = self.batch_size,
                                                                      topic_size = LTC_N_TOPIC_TEXT,
                                                                      memory_dim = self.final_encoder_dimension,
                                                                      input_hidden_dim = self.final_encoder_dimension,
                                                                      input_encoder = self.final_encoder,
                                                                      dr_memory_prob= self.dr_text_ltc_ph
                                                                     )

        
    def _create_output_layers(self):
        print ('[launch-text] create output projection layer')
        
        with tf.name_scope('text_output_layer') as scope:

            self.M = tf.Variable(tf.random.uniform([self.final_encoder_dimension, N_CATEGORY],
                                                   minval= -0.25,
                                                   maxval= 0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                                 trainable=True,
                                                 name="similarity_matrix")
            
            self.b = tf.Variable(tf.zeros([N_CATEGORY], dtype=tf.float32),
                                                 trainable=True,
                                                 name="output_bias")
            
            # e * M + b
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b
        
        with tf.name_scope('loss') as scope:
            
            self.batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.batch_pred, labels=self.y_labels )
            self.loss = tf.reduce_mean( self.batch_loss  )
                    
                
    def _create_output_layers_for_multi(self):
        print ('[launch-text] create output projection layer for multi')
        
        with tf.name_scope('text_output_layer') as scope:

            """
            self.M = tf.Variable(tf.random.uniform([self.final_encoder_dimension, (self.final_encoder_dimension/2)],
                                                   minval= -0.25,
                                                   maxval= 0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                                 trainable=True,
                                                 name="similarity_matrix")
            
            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                                 trainable=True,
                                                 name="output_bias")
            
            # e * M + b
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b
            """
            #self.batch_pred = self.final_encoder
        
    
    def _create_optimizer(self):
        print ('[launch-text] create optimizer')
        
        with tf.name_scope('text_optimizer') as scope:
            opt_func = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
            gvs = opt_func.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_norm(t=grad, clip_norm=1), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)
    
    
    def _create_summary(self):
        print ('[launch-text] create summary')
        
        with tf.name_scope('summary'):
            tf.compat.v1.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.compat.v1.summary.merge_all()
    
    
    def build_graph(self):
        print('[object created] ', self.__class__.__name__)
        
        self._create_placeholders()
        self._create_embedding()
        self._use_external_embedding()
        self._create_gru_model() 
        
        if self.attn      : self._add_attn()
        if self.ltc : self._add_LTC_method()
        
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()
        
        
        
    def build_graph_multi(self):
        print('[object created] ', self.__class__.__name__)
        
        self._create_placeholders()
        self._create_embedding()
        self._use_external_embedding()
        self._create_gru_model() 
        
        if self.attn      : self._add_attn()
        if self.ltc :  self._add_LTC_method()
