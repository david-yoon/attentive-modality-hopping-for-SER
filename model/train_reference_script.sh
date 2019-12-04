###########################################
# SE train - three modality
# encoder_size_audio = 750
# encoder_size_text  = 128
# control = hop
###########################################
CUDA_VISIBLE_DEVICES=0 python train_AMH.py --batch_size 128 --lr 0.001 --encoder_size_audio 750 --num_layer_audio 1 --hidden_dim_audio 200 --dr_audio 0.7 --bi_audio 0 --encoder_size_text 128 --num_layer_text 1 --hidden_dim_text 200 --dr_text 0.3 --bi_text 0 --encoder_size_video 25 --num_layer_video 1 --hidden_dim_video 128 --dr_video 0.7 --bi_video 0 --num_train_steps 10000 --is_save 0  --use_glove 1 --graph_prefix 'AMH' --data_path '../data/target_seven_120/fold01/' --hop 7