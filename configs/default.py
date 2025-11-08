from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.DROPOUT = 0.0
_C.MODEL.D_MODEL = 512
_C.MODEL.NUM_HEADS = 8
_C.MODEL.D_FF = 2048
_C.MODEL.ENCODER = 6
_C.MODEL.DECODER = 6
# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.LR = 1e-4
_C.TRAIN.NUM_EPOCHS = 60
_C.TRAIN.LOSS = 'CrossEntropy'
_C.TRAIN.SCHEDULER = True
_C.TRAIN.SAVECKPT_PERIOD = 10
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.SEED = 2

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = ('WikiText')
_C.DATASET.PATH = ('./data')
_C.DATASET.BPTT = 35
_C.DATASET.EN_DE = True

# ----------------------------------------------------------------------------
# Misc options
# ----------------------------------------------------------------------------
_C.OUTPUT_DIR = ""
_C.RESUME_CKPT = ""