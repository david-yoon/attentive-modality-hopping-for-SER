#-*- coding: utf-8 -*-

import tensorflow as tf

'''
desc : latent topic cluster method

input :
   - batch_size        : batch in input_encoder
   - topic             : # of topics
   - memory_dim        : dim of each topic
   - input_hidden_dim  : dim of input vector
   - input_encoder     : [batch, input_hidden_dim]
   - dr_memory_prob    : dropout ratio for memory

output : 
   - final_encoder : LTC applied vector [batch, vector(== concat(original, topic_mem)]
   - final_encoder_dimension : 
'''
def sy_ltc( batch_size, topic_size, memory_dim, input_hidden_dim, input_encoder, dr_memory_prob=1.0 ):
    print ('[launch : model_sy_ltc] s.y. Latent Topic Cluster method')
    
    with tf.name_scope('sy_ltc') as scope:

        with tf.variable_scope("sy_ltc", reuse=tf.AUTO_REUSE):
            # memory space for latent topic
            memory = tf.get_variable( "latent_topic_memory", 
                                          shape=[topic_size, memory_dim],
                                          initializer=tf.orthogonal_initializer()
                                         )

            memory_W = tf.Variable(tf.random_uniform( [input_hidden_dim, memory_dim],
                                                          minval= -0.25,
                                                          dtype=tf.float32,
                                                          seed=None),
                                        name="memory_projection_W")

        memory_W    = tf.nn.dropout( memory_W, keep_prob=dr_memory_prob )
        memory_bias = tf.Variable(tf.zeros([memory_dim], dtype=tf.float32), name="memory_projection_bias")
        
        # [batch_size, memory_dim]
        topic_sim_project = tf.matmul( input_encoder, memory_W ) + memory_bias
        
        # context 와 topic 의 similairty 계산 [batch, topic_size]
        topic_sim = tf.matmul( topic_sim_project, memory, transpose_b=True )

        # add non-linearity
        topic_sim = tf.tanh( topic_sim )

        # normalize [batch, topic_size]
        topic_sim_sigmoid_softmax = tf.nn.softmax( logits=topic_sim, dim=-1)

        # memory_context 를 계산  memory 를 topic_sim_norm 으로 weighted sum 수행
        # batch_size = 1 인 경우를 위해서 shape 을 맞추어줌
        # batch_size > 1 인 경우는 원래 형태와 변화가 없음
        shaped_input = tf.reshape( topic_sim_sigmoid_softmax, [batch_size, topic_size])

        # [batch, memory_dim, topic_size]
        topic_sim_mul_memory = tf.scan( lambda a, x : tf.multiply( tf.transpose(memory), x ), shaped_input, initializer=tf.transpose(memory) )
        
        tmpT = tf.reduce_sum(topic_sim_mul_memory, axis=-1, keep_dims=True)
        rsum = tf.squeeze( tmpT )
        
        #tmpT2 = tf.transpose(tmpT, [0, 2, 1])
        #rsum = tf.reshape( tmpT2, [batch_size, memory_dim])

        # final context 
        final_encoder  = tf.concat( [input_encoder, rsum], axis=-1 )

        final_encoder_dimension  = input_hidden_dim + memory_dim   # concat 으로 늘어났음

        return final_encoder, final_encoder_dimension
        
        
        
"""
same function as sy_ltc
return only weighted-memory
"""
def sy_ltc_memory_only( batch_size, topic_size, memory_dim, input_hidden_dim, input_encoder, dr_memory_prob=1.0 ):
    print ('[launch : model_sy_ltc] s.y. Latent Topic Cluster method')
    
    with tf.name_scope('sy_ltc') as scope:

        with tf.variable_scope("sy_ltc", reuse=tf.AUTO_REUSE):
            # memory space for latent topic
            memory = tf.get_variable( "latent_topic_memory", 
                                          shape=[topic_size, memory_dim],
                                          initializer=tf.orthogonal_initializer()
                                         )

            memory_W = tf.Variable(tf.random_uniform( [input_hidden_dim, memory_dim],
                                                          minval= -0.25,
                                                          dtype=tf.float32,
                                                          seed=None),
                                        name="memory_projection_W")

        memory_W    = tf.nn.dropout( memory_W, keep_prob=dr_memory_prob )
        memory_bias = tf.Variable(tf.zeros([memory_dim], dtype=tf.float32), name="memory_projection_bias")
        
        # [batch_size, memory_dim]
        topic_sim_project = tf.matmul( input_encoder, memory_W ) + memory_bias
        
        # context 와 topic 의 similairty 계산 [batch, topic_size]
        topic_sim = tf.matmul( topic_sim_project, memory, transpose_b=True )

        # add non-linearity
        topic_sim = tf.tanh( topic_sim )

        # normalize [batch, topic_size]
        topic_sim_sigmoid_softmax = tf.nn.softmax( logits=topic_sim, dim=-1)

        # memory_context 를 계산  memory 를 topic_sim_norm 으로 weighted sum 수행
        # batch_size = 1 인 경우를 위해서 shape 을 맞추어줌
        # batch_size > 1 인 경우는 원래 형태와 변화가 없음
        shaped_input = tf.reshape( topic_sim_sigmoid_softmax, [batch_size, topic_size])

        # [batch, memory_dim, topic_size]
        topic_sim_mul_memory = tf.scan( lambda a, x : tf.multiply( tf.transpose(memory), x ), shaped_input, initializer=tf.transpose(memory) )
        
        tmpT = tf.reduce_sum(topic_sim_mul_memory, axis=-1, keep_dims=True)
        rsum = tf.squeeze( tmpT )
        
        # final context 
        final_encoder  = rsum
        final_encoder_dimension  = memory_dim

        return final_encoder, final_encoder_dimension