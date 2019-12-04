#-*- coding: utf-8 -*-

"""
what    : Single Encoder Model for Multi (Audio + Text + Video) with Attentive Modality Hopping
data    : IEMOCAP
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper 

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
import sys
from project_config import *
from layers import add_GRU

from model_audio import *
from model_text import *
from model_video import *

from model_luong_attention import luong_attention


class ModelAMH:

    def __init__(self,
                 batch_size,
                 lr,
                 encoder_size_audio,  # for audio
                 num_layer_audio,
                 hidden_dim_audio,
                 dr_audio,
                 bi_audio, attn_audio, ltc_audio,
                 dic_size,             # for text
                 use_glove,
                 encoder_size_text,
                 num_layer_text,
                 hidden_dim_text,
                 dr_text,
                 bi_text, attn_text, ltc_text,
                 encoder_size_video,  # for video
                 num_layer_video,
                 hidden_dim_video,
                 dr_video,
                 bi_video, attn_video, ltc_video,
                 type_modality=4,
                 hop=1
                ):

        # for audio
        self.encoder_size_audio = encoder_size_audio
        self.num_layers_audio = num_layer_audio
        self.hidden_dim_audio = hidden_dim_audio
        self.dr_audio  = dr_audio
        
        self.encoder_inputs_audio = []
        self.encoder_seq_length_audio = []
        
        self.bi_audio = bi_audio
        self.attn_audio = attn_audio
        self.ltc_audio = ltc_audio
        
        # for text        
        self.dic_size = dic_size
        self.use_glove = use_glove
        self.encoder_size_text = encoder_size_text
        self.num_layers_text = num_layer_text
        self.hidden_dim_text = hidden_dim_text
        self.dr_text  = dr_text
        
        self.encoder_inputs_text = []
        self.encoder_seq_length_text =[]
                
        self.bi_text = bi_text
        self.attn_text = attn_text
        self.ltc_text = ltc_text
        
        self.hop = int(hop)

        
        # for video
        self.encoder_size_video = encoder_size_video
        self.num_layers_video = num_layer_video
        self.hidden_dim_video = hidden_dim_video
        self.dr_video  = dr_video
        
        self.encoder_inputs_video = []
        self.encoder_seq_length_video = []
        
        self.bi_video = bi_video
        self.attn_video = attn_video
        self.ltc_video = ltc_video
        
        
        # common        
        self.batch_size = batch_size
        self.lr = lr
        self.type_modality = type_modality
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


    def _create_model_audio_text_video(self):
        print ('[launch-multi] create audio/text/video model')
        
        # Audio
        self.model_audio =  ModelAudio(
                                                        batch_size=self.batch_size,
                                                        encoder_size=self.encoder_size_audio,
                                                        num_layer=self.num_layers_audio,
                                                        hidden_dim=self.hidden_dim_audio,
                                                        lr = self.lr,
                                                        dr = self.dr_audio,
                                                        bi = self.bi_audio,
                                                        attn = self.attn_audio,
                                                        ltc = self.ltc_audio
                                                        )
        self.model_audio.build_graph_multi()
        
        # Text
        self.model_text = ModelText(
                                                        batch_size=self.batch_size,
                                                        dic_size=self.dic_size,
                                                        use_glove=self.use_glove,
                                                        encoder_size=self.encoder_size_text,
                                                        num_layer=self.num_layers_text,
                                                        hidden_dim=self.hidden_dim_text,
                                                        lr = self.lr,
                                                        dr= self.dr_text,
                                                        bi = self.bi_text,
                                                        attn = self.attn_text,
                                                        ltc = self.ltc_text
                                                        )
        
        self.model_text.build_graph_multi()

        # Video
        self.model_video =  ModelVideo(
                                                        batch_size=self.batch_size,
                                                        encoder_size=self.encoder_size_video,
                                                        num_layer=self.num_layers_video,
                                                        hidden_dim=self.hidden_dim_video,
                                                        lr = self.lr,
                                                        dr = self.dr_video,
                                                        bi = self.bi_video,
                                                        attn = self.attn_video,
                                                        ltc = self.ltc_video
                                                        )
        self.model_video.build_graph_multi()
        
        
    def _create_placeholders(self):
        print ('[launch-multi] placeholders')
        with tf.name_scope('multi_placeholder'):
            
            # for audio
            self.encoder_inputs_audio  = self.model_audio.encoder_inputs  # [batch, time_step, audio]
            self.encoder_seq_audio     = self.model_audio.encoder_seq
            self.encoder_prosody       = self.model_audio.encoder_prosody
            self.dr_audio_in_ph        = self.model_audio.dr_audio_in_ph
            self.dr_audio_out_ph       = self.model_audio.dr_audio_out_ph
            
            # for text
            self.encoder_inputs_text  = self.model_text.encoder_inputs
            self.encoder_seq_text     = self.model_text.encoder_seq
            self.dr_text_in_ph        = self.model_text.dr_text_in_ph
            self.dr_text_out_ph       = self.model_text.dr_text_out_ph

            # for video
            self.encoder_inputs_video  = self.model_video.encoder_inputs  # [batch, time_step, audio]
            self.encoder_seq_video     = self.model_video.encoder_seq
            self.dr_video_in_ph        = self.model_video.dr_video_in_ph
            self.dr_video_out_ph       = self.model_video.dr_video_out_ph
            
            # common
            self.y_labels             = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size, N_CATEGORY], name="label")
            
            # for using pre-trained embedding
            self.embedding_placeholder = self.model_text.embedding_placeholder
            
        
    def _set_modality(self):
        
        print("all the three modalities are used")
        self.modality_1_final_encoder_dimension = self.model_audio.final_encoder_dimension
        self.modality_1_final_encoder           = self.model_audio.final_encoder
        self.modality_1_outputs                 = self.model_audio.outputs
        self.modality_1_encoder_seq             = self.model_audio.encoder_seq
        self.modality_1_encoder_size            = self.encoder_size_audio


        self.modality_2_final_encoder_dimension = self.model_text.final_encoder_dimension
        self.modality_2_final_encoder           = self.model_text.final_encoder
        self.modality_2_outputs                 = self.model_text.outputs
        self.modality_2_encoder_seq             = self.model_text.encoder_seq
        self.modality_2_encoder_size            = self.encoder_size_text
        
        
        self.modality_3_final_encoder_dimension = self.model_video.final_encoder_dimension
        self.modality_3_final_encoder           = self.model_video.final_encoder
        self.modality_3_outputs                 = self.model_video.outputs
        self.modality_3_encoder_seq             = self.model_video.encoder_seq
        self.modality_3_encoder_size            = self.encoder_size_video

            
    
    def _create_attention_layers_type_1(self, name):
        print ('[launch-multi-attn] create an attention layer: A(1)+T(2) --> V(3)')
        
        with tf.name_scope('attention_layer_' + str(name)) as scope:
        
            attnM = tf.Variable(tf.random.uniform([self.modality_1_final_encoder_dimension + self.modality_2_final_encoder_dimension,
                                                        self.modality_3_final_encoder_dimension],
                                                       minval= -0.25,
                                                       maxval= 0.25,
                                                       dtype=tf.float32,
                                                       seed=None),
                                                     trainable=True,
                                                     name="attn_projection_helper")

            attnb = tf.Variable(tf.zeros([self.modality_3_final_encoder_dimension], dtype=tf.float32),
                                                     trainable=True,
                                                     name="attn_bias")

            query_prj = tf.matmul( tf.concat([self.modality_1_final_encoder, self.modality_2_final_encoder], axis=1) , attnM) + attnb
            
            
            target                = self.modality_3_outputs
            condition             = query_prj
            batch_seq             = self.modality_3_encoder_seq
            max_len               = self.modality_3_encoder_size
            hidden_dim            = self.modality_3_final_encoder_dimension
            
            self.modality_3_final_encoder, self.attn_norm_1hop = luong_attention( 
                                                                batch_size = self.batch_size,
                                                                target     = target,
                                                                condition  = condition,
                                                                batch_seq  = batch_seq,
                                                                max_len    = max_len,
                                                                hidden_dim = hidden_dim
                                                            )
            
            # apply weighted sum result to final_encoder
            attn_vector_concat    = tf.concat( [self.modality_1_final_encoder, self.modality_2_final_encoder, self.modality_3_final_encoder], axis=1 )
            
            # set fianl information
            self.final_encoder           = attn_vector_concat
            self.final_encoder_dimension = self.modality_1_final_encoder_dimension + self.modality_2_final_encoder_dimension + self.modality_3_final_encoder_dimension
            

    def _create_attention_layers_type_2(self, name):
        print ('[launch-multi-attn] create an attention layer: T(2)+V2(3) --> A(1)')
        
        with tf.name_scope('attention_layer_' + str(name)) as scope:
        
            # for audio case
            # audio_outputs [ batch, encoder_size, hidden_dim ]   - pick this!
            # final_encoder_dimension = hidden_dim + prosody_dim  - Not
            attnM = tf.Variable(tf.random.uniform([self.modality_2_final_encoder_dimension+self.modality_3_final_encoder_dimension,
                                                             self.modality_1_final_encoder_dimension],
                                                       minval= -0.25,
                                                       maxval= 0.25,
                                                       dtype=tf.float32,
                                                       seed=None),
                                                     trainable=True,
                                                     name="attn_projection_helper")

            attnb = tf.Variable(tf.zeros([self.modality_1_final_encoder_dimension], dtype=tf.float32),
                                                     trainable=True,
                                                     name="attn_bias")

            query_prj = tf.matmul( tf.concat([self.modality_2_final_encoder, self.modality_3_final_encoder], axis=1) , attnM) + attnb
            
            target                = self.modality_1_outputs
            condition             = query_prj
            batch_seq             = self.modality_1_encoder_seq
            max_len               = self.modality_1_encoder_size
            hidden_dim            = self.modality_1_final_encoder_dimension
            
            self.modality_1_final_encoder, self.attn_norm_2hop = luong_attention( 
                                                                batch_size = self.batch_size,
                                                                target     = target,
                                                                condition  = condition,
                                                                batch_seq  = batch_seq,
                                                                max_len    = max_len,
                                                                hidden_dim = hidden_dim
                                                            )
            
            # apply weighted sum result to final_encoder
            attn_vector_concat    = tf.concat( [self.modality_1_final_encoder, self.modality_2_final_encoder, self.modality_3_final_encoder], axis=1 )
            
            # set fianl information
            self.final_encoder           = attn_vector_concat
            self.final_encoder_dimension = self.modality_1_final_encoder_dimension + self.modality_2_final_encoder_dimension + self.modality_3_final_encoder_dimension
    
    
    
    def _create_attention_layers_type_3(self, name):
        print ('[launch-multi-attn] create an attention layer: A2(1)+V2(3) --> T(2)')
        
        with tf.name_scope('attention_layer_' + str(name)) as scope:
        
            # for audio case
            # audio_outputs [ batch, encoder_size, hidden_dim ]   - pick this!
            # final_encoder_dimension = hidden_dim + prosody_dim  - Not
            attnM = tf.Variable(tf.random.uniform([self.modality_1_final_encoder_dimension + self.modality_3_final_encoder_dimension,
                                                 self.modality_2_final_encoder_dimension],
                                           minval= -0.25,
                                           maxval= 0.25,
                                           dtype=tf.float32,
                                           seed=None),
                                         trainable=True,
                                         name="attn_projection_helper")

            attnb = tf.Variable(tf.zeros([self.modality_2_final_encoder_dimension], dtype=tf.float32),
                                                     trainable=True,
                                                     name="attn_bias")

            query_prj = tf.matmul( tf.concat([self.modality_1_final_encoder, self.modality_3_final_encoder], axis=1) , attnM) + attnb
            
            target                = self.modality_2_outputs
            condition             = query_prj
            batch_seq             = self.modality_2_encoder_seq
            max_len               = self.modality_2_encoder_size
            hidden_dim            = self.modality_2_final_encoder_dimension

            self.modality_2_final_encoder, self.attn_norm_3hop = luong_attention( 
                                                                batch_size = self.batch_size,
                                                                target     = target,
                                                                condition  = condition,
                                                                batch_seq  = batch_seq,
                                                                max_len    = max_len,
                                                                hidden_dim = hidden_dim
                                                            )

            
            # apply weighted sum result to final_encoder
            attn_vector_concat    = tf.concat( [self.modality_1_final_encoder, self.modality_2_final_encoder, self.modality_3_final_encoder], axis=1 )
            
            # set fianl information
            self.final_encoder           = attn_vector_concat
            self.final_encoder_dimension = self.modality_1_final_encoder_dimension + self.modality_2_final_encoder_dimension + self.modality_3_final_encoder_dimension
    
    
    def _add_LTC_method(self):
        from model_sy_ltc import sy_ltc
        print ('[launch-text] apply LTC method')

        with tf.name_scope('text_LTC') as scope:
            self.final_encoder, self.final_encoder_dimension = sy_ltc( batch_size = self.batch_size,
                                                                      topic_size = LTC_N_TOPIC_MULTI,
                                                                      memory_dim = LTC_MEM_DIM_MULTI,
                                                                      input_hidden_dim = self.final_encoder_dimension,
                                                                      input_encoder = self.final_encoder,
                                                                      dr_memory_prob= LTC_DR_MULTI
                                                                     )
            
    
            
    def _add_prosody(self):
        print ('[launch-audio] add prosody feature, dim: ' + str(N_AUDIO_PROSODY))
        self.final_encoder           = tf.concat( [self.final_encoder, self.model_audio.encoder_prosody], axis=1 )
        self.final_encoder_dimension = self.final_encoder_dimension + N_AUDIO_PROSODY

        
        
    def _create_output_layers(self):
        print ('[launch-multi] create output projection layer')
        
        with tf.name_scope('multi_output_layer') as scope:
            
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

            
    def _create_output_layers_ff(self):
        print ('[launch-multi] create output projection layer FF')
        
        with tf.name_scope('multi_output_layer_ff') as scope:

            self.final_encoder = tf.concat( [self.model_audio.batch_pred, self.model_text.batch_pred], axis=1 )
            
            initializers = tf.contrib.layers.xavier_initializer(
                                                            uniform=True,
                                                            seed=None,
                                                            dtype=tf.float32
                                                            )
            
            self.batch_pred = tf.contrib.layers.fully_connected( 
                                                            inputs = self.final_encoder,
                                                            num_outputs = N_CATEGORY,
                                                            activation_fn = tf.nn.tanh,
                                                            normalizer_fn=None,
                                                            normalizer_params=None,
                                                            weights_initializer=initializers,
                                                            weights_regularizer=None,
                                                            biases_initializer=tf.zeros_initializer(),
                                                            biases_regularizer=None,
                                                            trainable=True
                                                            )
            
        
        with tf.name_scope('loss') as scope:
            self.batch_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.batch_pred, labels=self.y_labels )
            self.loss = tf.reduce_mean( self.batch_loss  )
            
            
    def _create_optimizer(self):
        print ('[launch-multi] create optimizer')
        
        with tf.name_scope('multi_optimizer') as scope:
            opt_func = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
            gradients, variables = zip(*opt_func.compute_gradients(self.loss))
            gradients = [None if gradient is None else tf.clip_by_norm(t=gradient, clip_norm=1.0) for gradient in gradients]
            self.optimizer = opt_func.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            
            
    def _create_summary(self):
        print ('[launch-multi] create summary')
        
        with tf.name_scope('summary'):
            tf.compat.v1.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.compat.v1.summary.merge_all()
            
            
    def build_graph(self):
        print('[object created] ', self.__class__.__name__)
        
        self._create_model_audio_text_video()
        self._create_placeholders()
        
        self._set_modality()
        self._create_attention_layers_type_1('hop1')
        if self.hop > 1: self._create_attention_layers_type_2('hop2')
        if self.hop > 2: self._create_attention_layers_type_3('hop3')
        if self.hop > 3: self._create_attention_layers_type_1('hop4')
        if self.hop > 4: self._create_attention_layers_type_2('hop5')
        if self.hop > 5: self._create_attention_layers_type_3('hop6')
        if self.hop > 6: self._create_attention_layers_type_1('hop7')
        if self.hop > 7: self._create_attention_layers_type_2('hop8')
        if self.hop > 8: self._create_attention_layers_type_3('hop9')
        if self.hop > 9: self._create_attention_layers_type_1('hop10')
        if self.hop > 10: self._create_attention_layers_type_2('hop11')
        if self.hop > 11: self._create_attention_layers_type_3('hop12')
        if self.hop > 12: 
            print('[ERROR] Not Implemented')
            sys.exit()
        
        if APPLY_LTC_MULTI : self._add_LTC_method()
            
        if USE_FF    : self._create_output_layers_ff()
        else         : self._create_output_layers()
        
        self._create_optimizer()
        self._create_summary()        