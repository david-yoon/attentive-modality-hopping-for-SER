#-*- coding: utf-8 -*-

"""
what    : train AMH model
data    : IEMOCAP
"""
import tensorflow as tf
import os
import time
import argparse
import datetime

from model_AMH import *
from process_data_AMH import *
from evaluation_AMH import *
from project_config import *


# for training         
def train_step(sess, model, batch_gen):
    raw_encoder_inputs_audio, raw_encoder_seq_audio, raw_encoder_prosody, raw_encoder_inputs_text, raw_encoder_seq_text, raw_encoder_inputs_video, raw_encoder_seq_video, raw_label = batch_gen.get_batch(
                            batch_size=model.batch_size,
                            data=batch_gen.train_set,
                            encoder_size_audio=model.encoder_size_audio,
                            encoder_size_text=model.encoder_size_text,
                            encoder_size_video=model.encoder_size_video,
                            is_test=False
                            )

    # prepare data which will be push from pc to placeholder
    input_feed = {}
    
    # Audio
    input_feed[model.encoder_inputs_audio] = raw_encoder_inputs_audio
    input_feed[model.encoder_seq_audio]    = raw_encoder_seq_audio
    input_feed[model.encoder_prosody]      = raw_encoder_prosody
    input_feed[model.dr_audio_in_ph]       = model.dr_audio
    input_feed[model.dr_audio_out_ph]      = 1.0
    
    # Text
    input_feed[model.encoder_inputs_text] = raw_encoder_inputs_text
    input_feed[model.encoder_seq_text]    = raw_encoder_seq_text
    input_feed[model.dr_text_in_ph]       = model.dr_text
    input_feed[model.dr_text_out_ph]      = 1.0
    
    # Video
    input_feed[model.encoder_inputs_video] = raw_encoder_inputs_video
    input_feed[model.encoder_seq_video]    = raw_encoder_seq_video
    input_feed[model.dr_video_in_ph]       = model.dr_video
    input_feed[model.dr_video_out_ph]      = 1.0
    
    # Common
    input_feed[model.y_labels] = raw_label
    
    _, summary = sess.run([model.optimizer, model.summary_op], input_feed)
    
    return summary

    
def train_model(model, batch_gen, num_train_steps, valid_freq, is_save=0, graph_dir_name='default'):
    
    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    
    summary = None
    val_summary = None
    
    with tf.compat.v1.Session(config=config) as sess:
        
        sess.run(tf.compat.v1.global_variables_initializer())
        early_stop_count = MAX_EARLY_STOP_COUNT
        
        # for text model - GLOVE
        if model.use_glove == 1:
            sess.run(model.model_text.embedding_init, feed_dict={ model.embedding_placeholder: batch_gen.get_glove() })
            print ('[completed] loading pre-trained embedding vector to placeholder')
            
        
        # if exists check point, starts from the check point
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('save/' + graph_dir_name + '/'))
        if ckpt and ckpt.model_checkpoint_path:
            print ('from check point!!!')
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        writer = tf.compat.v1.summary.FileWriter('./graph/'+graph_dir_name, sess.graph)

        initial_time = time.time()
        
        min_ce = 1000000
        target_best    = 0
        target_dev_WA  = 0
        target_dev_UA  = 0
        target_test_WA = 0
        target_test_UA = 0
        
        for index in range(num_train_steps):

            try:
                # run train 
                summary = train_step(sess, model, batch_gen)
                writer.add_summary( summary, global_step=model.global_step.eval() )
                
            except:
                print ("excepetion occurs in train step")
                pass
                
            
            # run validation
            if (index + 1) % valid_freq == 0:
                
                dev_ce, dev_WA, dev_UA, dev_summary, _ = run_test(sess=sess,
                                                         model=model, 
                                                         batch_gen=batch_gen,
                                                         data=batch_gen.dev_set)
                
                writer.add_summary( dev_summary, global_step=model.global_step.eval() )
                
                end_time = time.time()

                if index > CAL_ACCURACY_FROM:
                    
                    if IS_TRAGET_OBJECTIVE_WA : target = dev_WA
                    else                      : target = dev_UA
                                

                    if ( target > target_best ):
                        target_best = target

                        # save best result
                        if is_save is 1:
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval() )
                            
                            
                        if float(target) > float(QUICK_SAVE_THRESHOLD):
                            print("[INFO] apply quick save")
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval() )
                            

                        early_stop_count = MAX_EARLY_STOP_COUNT
                        
                        test_ce, test_WA, test_UA, _, _ = run_test(sess=sess,
                                                         model=model,
                                                         batch_gen=batch_gen,
                                                         data=batch_gen.test_set)

                        target_dev_WA  = dev_WA
                        target_dev_UA  = dev_UA
                        target_test_WA = test_WA
                        target_test_UA = test_UA
                                                                                            
                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print ("early stopped")
                            break
                             
                        early_stop_count = early_stop_count -1
                        
                    print (str( int((end_time - initial_time)/60) ) + " mins" + \
                        " step/seen/itr: " + str( model.global_step.eval() ) + "/ " + \
                                               str( model.global_step.eval() * model.batch_size ) + "/" + \
                                               str( round( model.global_step.eval() * model.batch_size / float(len(batch_gen.train_set)), 2)  ) + \
                        "\t(dev WA/UA): " + \
                        '{:.3f}'.format(dev_WA)  + '\t'  + \
                        '{:.3f}'.format(dev_UA)  + '\t' + \
                        "(test WA/UA): "  + \
                        '{:.3f}'.format(test_WA) + '\t'  + \
                        '{:.3f}'.format(test_UA) + '\t' + \
                        " loss: " + '{:.2f}'.format(dev_ce))
                
        writer.close()
            
        print ('Total steps : {}'.format(model.global_step.eval()) )
        
        print ('final result at best step \t' + \
                    '{:.3f}'.format(target_dev_WA) + '\t' + \
                    '{:.3f}'.format(target_dev_UA) + '\t' + \
                    '{:.3f}'.format(target_test_WA) + '\t' + \
                    '{:.3f}'.format(target_test_UA) + \
                    '\n')

        # result logging to file
        with open('./TEST_run_result.txt', 'a') as f:
            f.write('\n' + \
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' + \
                    batch_gen.data_path.split('/')[-2] + '\t' + \
                    graph_dir_name + '\t' + \
                    '{:.3f}'.format(target_dev_WA) + '\t' + \
                    '{:.3f}'.format(target_dev_UA) + '\t' + \
                    '{:.3f}'.format(target_test_WA) + '\t' + \
                    '{:.3f}'.format(target_test_UA) )


def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
        
def main(data_path, batch_size, lr, type_modality,
         encoder_size_audio, num_layer_audio, hidden_dim_audio, dr_audio,
         bi_audio, attn_audio, ltc_audio,
         encoder_size_text, num_layer_text, hidden_dim_text, dr_text,
         use_glove,
         bi_text, attn_text, ltc_text,
         encoder_size_video, num_layer_video, hidden_dim_video, dr_video,
         bi_video, attn_video, ltc_video,
         hop,
         num_train_steps, is_save, graph_dir_name
         ):

    create_dir('save/')
    
    if is_save is 1:
        create_dir('save/'+ graph_dir_name )
    
    create_dir('graph/')
    create_dir('graph/'+graph_dir_name)
    
    batch_gen = ProcessDataAMH(data_path)
    
    model = ModelAMH(
                            batch_size = batch_size,
                            lr = lr,
                            type_modality = type_modality,
                            hop=hop,
                            encoder_size_audio = encoder_size_audio,      # Audio
                            num_layer_audio = num_layer_audio,
                            hidden_dim_audio = hidden_dim_audio,                
                            dr_audio = dr_audio,
                            bi_audio = bi_audio,
                            attn_audio = attn_audio,
                            ltc_audio = ltc_audio,
                            encoder_size_text = encoder_size_text,        # Text
                            num_layer_text = num_layer_text,
                            hidden_dim_text = hidden_dim_text,                
                            dr_text = dr_text,
                            dic_size = batch_gen.dic_size,
                            use_glove = use_glove,
                            bi_text = bi_text,
                            attn_text = attn_text, 
                            ltc_text = ltc_text,
                            encoder_size_video = encoder_size_video,    # Video
                            num_layer_video = num_layer_video,
                            hidden_dim_video = hidden_dim_video,                
                            dr_video = dr_video,
                            bi_video = bi_video,
                            attn_video = attn_video,
                            ltc_video = ltc_video
                            )

    model.build_graph()
    
    valid_freq = int( len(batch_gen.train_set) * EPOCH_PER_VALID_FREQ / float(batch_size)  ) + 1
    print ("[Info] Valid Freq = " + str(valid_freq))

    train_model(model, batch_gen, num_train_steps, valid_freq, is_save, graph_dir_name)
    
if __name__ == '__main__':

    # Common
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_train_steps', type=int, default=10000)
    p.add_argument('--is_save', type=int, default=0)
    p.add_argument('--graph_prefix', type=str, default="default")
    p.add_argument('--mType', type=int, default=4)
    p.add_argument('--hop', type=int, default=1)
    
    # Audio
    p.add_argument('--encoder_size_audio', type=int, default=750)
    p.add_argument('--num_layer_audio', type=int, default=1)
    p.add_argument('--hidden_dim_audio', type=int, default=50)
    p.add_argument('--dr_audio', type=float, default=1.0)
    p.add_argument('--bi_audio', type=int, default=0)
    p.add_argument('--attn_audio', type=int, default=0)
    p.add_argument('--ltc_audio', type=int, default=0)
    
    # Text
    p.add_argument('--use_glove', type=int, default=0)
    p.add_argument('--encoder_size_text', type=int, default=750)
    p.add_argument('--num_layer_text', type=int, default=1)
    p.add_argument('--hidden_dim_text', type=int, default=50)
    p.add_argument('--dr_text', type=float, default=1.0)
    p.add_argument('--bi_text', type=int, default=0)
    p.add_argument('--attn_text', type=int, default=0)
    p.add_argument('--ltc_text', type=int, default=0)
    
    # Video
    p.add_argument('--encoder_size_video', type=int, default=25)
    p.add_argument('--num_layer_video', type=int, default=1)
    p.add_argument('--hidden_dim_video', type=int, default=128)
    p.add_argument('--dr_video', type=float, default=1.0) 
    p.add_argument('--bi_video', type=int, default=0)
    p.add_argument('--attn_video', type=int, default=0)
    p.add_argument('--ltc_video', type=int, default=0)
    

    embed_train = ''
    if EMBEDDING_TRAIN == False:
        embed_train = 'F'
    
    args = p.parse_args()
    
    graph_name = args.graph_prefix + \
                    '_hop' + str(args.hop) + \
                    '_D' + (args.data_path).split('/')[-2] + \
                    '_b' + str(args.batch_size) + \
                    '_mType' + str(args.mType) + \
                    '_esA' + str(args.encoder_size_audio) + \
                    '_LA' + str(args.num_layer_audio) + \
                    '_HA' + str(args.hidden_dim_audio) + \
                    '_drA' + str(args.dr_audio) + \
                    '_biA' + str(args.bi_audio) + \
                    '_attnA' + str(args.attn_audio) + \
                    '_esT' + str(args.encoder_size_text) + \
                    '_LT' + str(args.num_layer_text) + \
                    '_HT' + str(args.hidden_dim_text) + \
                    '_G' + str(args.use_glove) + embed_train + \
                    '_drT' + str(args.dr_text) + \
                    '_biT' + str(args.bi_text) + \
                    '_attnT' + str(args.attn_text) + \
                    '_LV' + str(args.num_layer_video) + \
                    '_HV' + str(args.hidden_dim_video) + \
                    '_drV' + str(args.dr_video) + \
                    '_biV' + str(args.bi_video) + \
                    '_attnV' + str(args.attn_video)
    
    
    if args.ltc_audio == 1:
        graph = graph + '_ltcA' + str(args.ltc_audio)
    
    if args.ltc_text == 1:
        graph = graph + '_ltcT' + str(args.ltc_text)
    
    if args.ltc_video == 1:
        graph = graph + '_ltcV' + str(args.ltc_video)
    
    
    if  IS_AUDIO_RESIDUAL:
        print('[INFO] audio-residual')
        graph_name = graph_name + '_aR'

    if  IS_TEXT_RESIDUAL:
        print('[INFO] text-residual')
        graph_name = graph_name + '_tR'
        
    if  IS_VIDEO_RESIDUAL:
        print('[INFO] video-residual')
        graph_name = graph_name + '_vR'
                    
    
    graph_name = graph_name + '_' + datetime.datetime.now().strftime("%m-%d-%H-%M")

    
    print('[INFO] data:\t\t', args.data_path)
    print('[INFO] batch:\t\t', args.batch_size)
    print('[INFO] #-class:\t\t', N_CATEGORY)
    print('[INFO] modality : use all the three modality')
    print('[INFO] hop:\t\t', args.hop)
    print('[INFO] lr:\t\t', args.lr)
    
    print('[INFO]-A encoder_size:\t', args.encoder_size_audio)
    print('[INFO]-A num_layer:\t', args.num_layer_audio)
    print('[INFO]-A hidden_dim:\t', args.hidden_dim_audio)
    print('[INFO]-A dr:\t\t', args.dr_audio)
    print('[INFO]-A bi:\t\t', args.bi_audio)
    print('[INFO]-A attn:\t\t', args.attn_audio)
    print('[INFO]-A ltc:\t\t', args.ltc_audio)
    
    print('[INFO]-T encoder_size:\t', args.encoder_size_text)
    print('[INFO]-T num_layer:\t', args.num_layer_text)
    print('[INFO]-T hidden_dim:\t', args.hidden_dim_text)
    print('[INFO]-T dr:\t\t', args.dr_text)
    print('[INFO]-T bi:\t\t', args.bi_text)
    print('[INFO]-T attn:\t\t', args.attn_text)
    print('[INFO]-T ltc:\t\t', args.ltc_text)
    
    print('[INFO]-V encoder_size:\t', args.encoder_size_video)
    print('[INFO]-V num_layer:\t', args.num_layer_video)
    print('[INFO]-V hidden_dim:\t', args.hidden_dim_video)
    print('[INFO]-V dr:\t\t', args.dr_video)
    print('[INFO]-V bi:\t\t', args.bi_video)
    print('[INFO]-V attn:\t\t', args.attn_video)
    print('[INFO]-V ltc:\t\t', args.ltc_video)
    
    
    main(
        data_path = args.data_path,
        batch_size = args.batch_size,
        type_modality = args.mType,
        lr = args.lr,
        encoder_size_audio = args.encoder_size_audio,   # Audio
        num_layer_audio = args.num_layer_audio,
        hidden_dim_audio = args.hidden_dim_audio,
        dr_audio = args.dr_audio,
        bi_audio=args.bi_audio,
        attn_audio=args.attn_audio,
        ltc_audio=args.ltc_audio,
        use_glove = args.use_glove,                    # Text
        encoder_size_text = args.encoder_size_text,
        num_layer_text = args.num_layer_text,
        hidden_dim_text = args.hidden_dim_text,
        dr_text = args.dr_text,
        bi_text =args.bi_text,
        attn_text =args.attn_text,
        ltc_text =args.ltc_text,
        encoder_size_video = args.encoder_size_video,  # Video
        num_layer_video = args.num_layer_video,
        hidden_dim_video = args.hidden_dim_video,
        dr_video = args.dr_video,
        bi_video=args.bi_video,
        attn_video=args.attn_video,
        ltc_video=args.ltc_video,
        hop=args.hop,
        num_train_steps = args.num_train_steps,
        is_save = args.is_save,
        graph_dir_name = graph_name
        )