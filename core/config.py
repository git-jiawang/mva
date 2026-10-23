from easydict import EasyDict as edict
import yaml

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            config[k][vk] = vv
                        else:
                            raise ValueError("{}.{} not exist in config.py".format(k, vk))
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

config = edict()

config.GPUS = '0'
config.USE_GPUS = '5'

config.IMG_SIZE = (224, 224)
config.DATASET = 'wildtrack'
config.VIEW_IDS = [0,1,2,3,4,5,6]
config.PROMPT_MODE = 'box'
config.TRAJECTORY_LEN = 3
config.CKP = ''
config.DEPTH = 2

# training
config.TRAIN = edict()
config.TRAIN.BATCH_SIZE = 1
config.TRAIN.EPOCH = 1200
config.TRAIN.RESUME = False
config.TRAIN.LR = 1e-4
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [900]
config.TRAIN.EDGE_ASSOCIATION = False
config.TRAIN.EDGE_LOSS_WEIGHT = 0.1

# testing
config.TEST = edict()
config.TEST.THRESH = 0.4

config.REID = False
config.ALPHA = 0.1

config.EXP_DIR = './results'










