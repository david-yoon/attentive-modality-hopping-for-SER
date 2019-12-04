
################################
#     Training             
################################
CAL_ACCURACY_FROM      = 0
MAX_EARLY_STOP_COUNT   = 15
EPOCH_PER_VALID_FREQ   = 0.3

DATA_TRAIN_MFCC        = 'train_audio_mfcc.npy'
DATA_TRAIN_MFCC_SEQN   = 'train_audio_seqN.npy'
DATA_TRAIN_PROSODY     = 'train_audio_prosody.npy'
DATA_TRAIN_LABEL       = 'train_label.npy'
DATA_TRAIN_TRANS       = 'train_nlp_trans.npy'
DATA_TRAIN_VIDEO        = 'train_video.npy'
DATA_TRAIN_VIDEO_SEQN   = 'train_video_seqN.npy'


DATA_DEV_MFCC          = 'dev_audio_mfcc.npy'
DATA_DEV_MFCC_SEQN     = 'dev_audio_seqN.npy'
DATA_DEV_PROSODY       = 'dev_audio_prosody.npy'
DATA_DEV_LABEL         = 'dev_label.npy'
DATA_DEV_TRANS         = 'dev_nlp_trans.npy'
DATA_DEV_VIDEO         = 'dev_video.npy'
DATA_DEV_VIDEO_SEQN    = 'dev_video_seqN.npy'


DATA_TEST_MFCC         = 'test_audio_mfcc.npy'
DATA_TEST_MFCC_SEQN    = 'test_audio_seqN.npy'
DATA_TEST_PROSODY      = 'test_audio_prosody.npy'
DATA_TEST_LABEL        = 'test_label.npy'
DATA_TEST_TRANS        = 'test_nlp_trans.npy'
DATA_TEST_VIDEO        = 'test_video.npy'
DATA_TEST_VIDEO_SEQN   = 'test_video_seqN.npy'

DIC                    = 'dic.pkl'
GLOVE                  = 'W_embedding.npy'

QUICK_SAVE_THRESHOLD   = 0.90
N_CATEGORY             = 7

################################
#     Audio
################################
N_AUDIO_MFCC    = 120    # 39 or 120 according to the dataset
N_AUDIO_PROSODY = 35


################################
#     Video
################################
N_SEQ_MAX_VIDEO = 32            #225
N_VIDEO         = 2048


################################
#     NLP
################################
N_SEQ_MAX_NLP      = 128   # normal 128, ASR 90
DIM_WORD_EMBEDDING = 100   # when using glove it goes to 300 automatically
EMBEDDING_TRAIN    = True  # True is better


################################
#     Multi
################################
USE_FF    = False


################################
#     Model
################################

IS_AUDIO_RESIDUAL      = False
IS_TEXT_RESIDUAL       = False
IS_VIDEO_RESIDUAL      = False

LTC_N_TOPIC_AUDIO = 2
LTC_MEM_DIM_AUDIO = 256
LTC_DR_AUDIO      = 0.8

LTC_N_TOPIC_VIDEO = 2
LTC_MEM_DIM_VIDEO = 256
LTC_DR_VIDEO      = 0.8

LTC_N_TOPIC_TEXT = 2
LTC_MEM_DIM_TEXT = 256
LTC_DR_TEXT      = 0.8

APPLY_LTC_MULTI   = False   # special case (controlled here)
LTC_N_TOPIC_MULTI = 2
LTC_MEM_DIM_MULTI = 256
LTC_DR_MULTI      = 1.0


################################
#     MEASRE
# macro    (UA) : Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# weighted (WA) : Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).
################################
WA = 'weighted'
UA = 'macro'
IS_TRAGET_OBJECTIVE_WA = True


################################
#   ETC
################################
TEST_LOG_FOR_ANALYSIS = False