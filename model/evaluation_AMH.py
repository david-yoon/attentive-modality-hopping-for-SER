#-*- coding: utf-8 -*-

"""
what    : evaluation
data    : IEMOCAP
"""

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from project_config import *

"""
    desc  : 
    
    inputs: 
        sess  : tf session
        model : model for test
        data  : such as the dev_set, test_set...
            
    return:
        sum_batch_ce : sum cross_entropy
        accr         : accuracy
        
"""
def run_test(sess, model, batch_gen, data):
    
    list_batch_ce = []
    list_batch_correct = []
    
    list_pred = []
    list_label = []

    max_loop  = int( len(data) / model.batch_size )

    # run 1 more time ( for batch remaining )
    # evaluate data ( N of chunk (batch_size) + remaining( +1) )
    for test_itr in range( max_loop + 1 ):
        
        raw_encoder_inputs_audio, raw_encoder_seq_audio, raw_encoder_prosody, raw_encoder_inputs_text, raw_encoder_seq_text, raw_encoder_inputs_video, raw_encoder_seq_video, raw_label = batch_gen.get_batch(
                            batch_size=model.batch_size,
                            data=data,
                            encoder_size_audio=model.encoder_size_audio,
                            encoder_size_text=model.encoder_size_text,
                            encoder_size_video=model.encoder_size_video,
                            is_test=True,
                            start_index= (test_itr* model.batch_size)
                            )
        
        # prepare data which will be push from pc to placeholder
        input_feed = {}

        # Audio
        input_feed[model.encoder_inputs_audio] = raw_encoder_inputs_audio
        input_feed[model.encoder_seq_audio]    = raw_encoder_seq_audio
        input_feed[model.encoder_prosody]      = raw_encoder_prosody
        input_feed[model.dr_audio_in_ph]       = 1.0
        input_feed[model.dr_audio_out_ph]      = 1.0

        # Text
        input_feed[model.encoder_inputs_text] = raw_encoder_inputs_text
        input_feed[model.encoder_seq_text]    = raw_encoder_seq_text
        input_feed[model.dr_text_in_ph]       = 1.0
        input_feed[model.dr_text_out_ph]      = 1.0

        # Video
        input_feed[model.encoder_inputs_video] = raw_encoder_inputs_video
        input_feed[model.encoder_seq_video]    = raw_encoder_seq_video
        input_feed[model.dr_video_in_ph]       = 1.0
        input_feed[model.dr_video_out_ph]      = 1.0
        
        # Common
        input_feed[model.y_labels] = raw_label
        
        try:
            bpred, bloss = sess.run([model.batch_pred, model.batch_loss], input_feed)
        except:
            print ("excepetion occurs in valid step : " + str(test_itr))
            pass
        
        # batch loss
        list_batch_ce.extend( bloss )
        
        # batch accuracy
        list_pred.extend( np.argmax(bpred, axis=1) )
        list_label.extend( np.argmax(raw_label, axis=1) )
        
    # cut-off dummy data used for batch
    list_pred  = list_pred[:len(data)]
    list_label = list_label[:len(data)]
    list_batch_ce = list_batch_ce[:len(data)] 
    
    
    # for log
    if TEST_LOG_FOR_ANALYSIS:
        with open( '../analysis/inference_log/mult_attn.txt', 'w' ) as f:
            f.write( ' '.join( [str(x) for x in list_pred] ) )

        with open( '../analysis/inference_log/mult_attn_label.txt', 'w' ) as f:
            f.write( ' '.join( [str(x) for x in list_label] ) )


    # macro : unweighted mean
    # weighted : ignore class unbalance
    accr_WA = precision_score(y_true=list_label,
                           y_pred=list_pred,
                           average=WA)
    
    accr_UA = precision_score(y_true=list_label,
                           y_pred=list_pred,
                           average=UA)
    
    sum_batch_ce = np.sum( list_batch_ce )
    
    value1 = summary_pb2.Summary.Value(tag="valid_loss", simple_value=sum_batch_ce)
    value2 = summary_pb2.Summary.Value(tag="valid_accuracy", simple_value=accr_WA )
    summary = summary_pb2.Summary(value=[value1, value2])
    
    return sum_batch_ce, accr_WA, accr_UA, summary, list_pred