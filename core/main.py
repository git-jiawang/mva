from config import config, update_config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = config.USE_GPUS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=所有, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import setproctitle
setproctitle.setproctitle("TA-MVA")

import torch
from torch.utils.data import DataLoader
from dataset import TrajectoryLoader
from model import MultiViewTrajectoryEncoder
from losses import *
from utils import *
import numpy as np
from sklearn.metrics import average_precision_score
import argparse
import pprint
from torchreid.utils import FeatureExtractor
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

def compute_loss(config, model, data, device):
    anchor_info = data[0]
    pos_info = data[1]
    anchor_trajectory = anchor_info['trajectories']
    pos_trajectory = pos_info['trajectories']
    if len(anchor_trajectory) == 0 or len(pos_trajectory) == 0:
        return None, None
    anchor_prompts = anchor_trajectory.squeeze(0).to(device)
    pos_prompts = pos_trajectory.squeeze(0).to(device)

    anchor_fea = model.module(anchor_info['cam_id'], (anchor_info['height'], anchor_info['width']), anchor_prompts)
    pos_fea = model.module(pos_info['cam_id'], (pos_info['height'], pos_info['width']), pos_prompts)

    labels1 = torch.cat(anchor_info['labels'], dim=0).to(device)
    labels2 = torch.cat(pos_info['labels'], dim=0).to(device)
    
    loss = triplet_loss(anchor_fea, pos_fea, labels1, labels2)
    
    if loss == -1:
        return None, None
        
    # 轨迹边损失
    edge_loss = 0
    if config.TRAIN.EDGE_ASSOCIATION:
        anchor_center_trajectory = anchor_info['center_trajectories']
        pos_center_trajectory = pos_info['center_trajectories']
        if len(anchor_center_trajectory) == 0 or len(pos_center_trajectory) == 0:
            return loss, edge_loss
        anchor_center_prompts = anchor_center_trajectory.squeeze(0).to(device)
        pos_center_prompts = pos_center_trajectory.squeeze(0).to(device)
        
        # 计算边的特征
        anchor_edge_fea = model.module(anchor_info['cam_id'], (anchor_info['height'], anchor_info['width']), anchor_center_prompts)
        pos_edge_fea = model.module(pos_info['cam_id'], (pos_info['height'], pos_info['width']), pos_center_prompts)
        
        anchor_edge_labels = anchor_info['center_labels']
        pos_edge_labels = pos_info['center_labels']
        anchor_flat = [item[0] for item in anchor_edge_labels]
        pos_flat = [item[0] for item in pos_edge_labels]
        # 将字符串标签映射为整数ID（保证anchor和pos使用相同映射）
        all_labels = set(anchor_flat + pos_flat)
        label_to_id = {label: idx for idx, label in enumerate(all_labels)}
        anchor_edge_ids = torch.tensor([label_to_id[l] for l in anchor_flat], device=device)
        pos_edge_ids = torch.tensor([label_to_id[l] for l in pos_flat], device=device)
        
        edge_triplet_loss = triplet_loss(anchor_edge_fea, pos_edge_fea, anchor_edge_ids, pos_edge_ids)

        if edge_triplet_loss != -1:
            edge_loss = edge_triplet_loss * config.TRAIN.EDGE_LOSS_WEIGHT
            loss += edge_loss
            
    return loss, edge_loss

def evaluate(config, model, dataloader_test, device, thresh=0.0, reid_model=None):
    correct_sum = 0
    precision_total_sum = 0
    recall_total_sum = 0
    scores = []
    labels = []
    IPAA_correct_sum = 0
    IPAA_total_sum = 0
    IPAA_80_sum = 0
    IPAA_90_sum = 0
    IPAA_100_sum = 0
    IPAA_img_sum = 0
    
    for _, data in enumerate(tqdm(dataloader_test, desc="Testing ")):
        view_num = len(data)
        feas = []
        if config.REID:
            reid_feas = []
            
        # 得到每个视角中的轨迹特征
        for i in range(view_num):
            info = data[i]
            trajectories = info['trajectories']  # [N, 3, 4]
            if len(trajectories) == 0:
                feas.append([])
                if config.REID:
                    reid_feas.append([])
                continue
            prompts = trajectories.squeeze(0).to(device)
            fea = model.module(info['cam_id'], (info['height'], info['width']), prompts)
            feas.append(fea)
                  
            if config.REID:
                imgs_input = info['cropped_imgs'].to(device).squeeze(0)
                reid_feature = reid_model(imgs_input)
                reid_feas.append(reid_feature)
        
        cam_pairs = []
        for i in range(view_num - 1):
            for j in range(i + 1, view_num):
                cam_pairs.append((i, j))

        for cam_pair in cam_pairs:
            view1, view2 = cam_pair
            anchor_fea = feas[view1]
            pos_fea = feas[view2]
            if len(anchor_fea) == 0 or len(pos_fea) == 0:
                continue
            anchor_info = data[view1]
            pos_info = data[view2]

            labels1 = torch.cat(anchor_info['labels'], dim=0).to(device)
            labels2 = torch.cat(pos_info['labels'], dim=0).to(device)
            
            if config.REID:
                result = cross_view_matching_evaluation(anchor_fea, pos_fea, labels1, labels2, reid_fea1=reid_feas[view1], 
                                                        reid_fea2=reid_feas[view2], thresh=thresh, alpha=config.ALPHA)
            else:
                result = cross_view_matching_evaluation(anchor_fea, pos_fea, labels1, labels2, thresh=thresh)

            scores.append(result['scores'])
            labels.append(result['labels'])
            correct_sum += result['correct']
            precision_total_sum += result['precision_total']
            recall_total_sum += result['recall_total']
            IPAA_correct_sum += result['IPAA_correct']
            IPAA_total_sum += result['IPAA_total']
            ratio = result['IPAA_correct'] / result['IPAA_total']
            IPAA_100_sum += (ratio == 1.)
            IPAA_90_sum += (ratio >= 0.9)
            IPAA_80_sum += (ratio >= 0.8)
            IPAA_img_sum += 1
    
    precision = correct_sum / precision_total_sum if precision_total_sum else 0
    recall = correct_sum / recall_total_sum

    labels = np.concatenate(labels)
    scores = np.concatenate(scores)
    ap = average_precision_score(labels, scores)
    fpr_95 = FPR_95(labels, scores)
    match_acc = IPAA_correct_sum / IPAA_total_sum
    IPAA_100 = IPAA_100_sum / IPAA_img_sum
    IPAA_90 = IPAA_90_sum / IPAA_img_sum
    IPAA_80 = IPAA_80_sum / IPAA_img_sum

    msg = ("AP: {0:.4f}\tFPR-95: {1:.4f}\tP: {2:.4f}\tR: {3:.4f}\tAccuracy: {4:.4f}\tIPAA 100: {5:.4f}\tIPAA 90: {6:.4f}\tIPAA 80: {7:.4f}".format(
            ap, fpr_95, precision, recall, match_acc, IPAA_100, IPAA_90, IPAA_80))
    logger.info(msg)
    
    plot_AP_curve(labels, scores, ap, final_output_dir)
            
def train():
    gpus = [int(i) for i in config.GPUS.split(",")]
    device = torch.device('cuda')

    dataset_train = TrajectoryLoader(config=config, mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=config.TRAIN.BATCH_SIZE, num_workers=6, pin_memory=True, shuffle=True)
    dataset_valid = TrajectoryLoader(config=config, mode='valid')
    dataloader_valid = DataLoader(dataset_valid, batch_size=config.TRAIN.BATCH_SIZE, num_workers=6, pin_memory=True, shuffle=False)

    view_num = dataset_train.get_view_num()
    model = MultiViewTrajectoryEncoder(view_num, depth=config.DEPTH, trajectory_len=config.TRAJECTORY_LEN, mode=config.PROMPT_MODE)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    reid_model = None
    if config.REID:
        reid_model = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='weights/osnet_ain_ms_d_c.pth.tar',
            device='cuda')
        
    # 恢复训练检查点
    if config.TRAIN.RESUME:
        checkpoint_path = os.path.join(final_output_dir, 'checkpoint_latest.pth.tar')
        ckp = torch.load(checkpoint_path)
        
        last_epoch = ckp['epoch']
        start_epoch = last_epoch + 1  # 从下一轮开始
        model.module.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        best_train_loss = ckp['best_train_loss']
        best_train_loss_epoch = ckp['best_train_loss_epoch']
        best_val_loss = ckp['best_val_loss']
        best_val_loss_epoch = ckp['best_val_loss_epoch']
        train_loss_history = ckp['train_loss_history']
        val_loss_history = ckp['val_loss_history']
        epoch_history = ckp['epoch_history']
        logger.info("=> loaded latest checkpoint")
    else:
        # 初始化指标
        last_epoch = -1
        start_epoch = 0
        best_train_loss = float('inf')
        best_train_loss_epoch = -1
        best_val_loss = float('inf')
        best_val_loss_epoch = -1
        train_loss_history = []
        val_loss_history = []
        epoch_history = []

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch=last_epoch)

    for epoch_i in range(start_epoch, config.TRAIN.EPOCH):
        logger.info("learning rate for this epoch {}".format(lr_scheduler.get_last_lr()))
        
        train_epoch_loss = 0
        train_center_loss = 0
        val_epoch_loss = 0
        val_center_loss = 0
        
        model.train()
        for step_i, data in enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch_i}")):
            loss, edge_loss = compute_loss(config, model, data, device)
            
            if loss is None:
                continue
            
            train_epoch_loss += loss.item()
            train_center_loss += edge_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for step_i, data in enumerate(tqdm(dataloader_valid, desc=f"Val ")):
                loss, edge_loss = compute_loss(config, model, data, device)

                if loss is None:
                    continue
                
                val_epoch_loss += loss.item()
                val_center_loss += edge_loss
        
        train_epoch_loss /= len(dataloader_train)
        train_center_loss /= len(dataloader_train)
        val_epoch_loss /= len(dataloader_valid)
        val_center_loss /= len(dataloader_valid)
        
        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)
        epoch_history.append(epoch_i)
        
        logger.info('Epoch: {} - Train loss: {:.6f}, Train Center loss: {:.6f}, Val loss: {:.6f}, Val Center loss: {:.6f}'.format(
            epoch_i, train_epoch_loss, train_center_loss, val_epoch_loss, val_center_loss))
        
        lr_scheduler.step()
        
        # 保存训练损失最小的模型
        if train_epoch_loss < best_train_loss:
            best_train_loss = train_epoch_loss
            best_train_loss_epoch = epoch_i
            
            best_train_loss_model_path = os.path.join(final_output_dir, 'model_best_train_loss.pth.tar')
            torch.save({'model': model.module.state_dict()}, best_train_loss_model_path)
            logger.info("=> saved best training loss model to {} ".format(best_train_loss_model_path))
            
        # 保存验证损失最小的模型
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_val_loss_epoch = epoch_i
            
            best_val_loss_model_path = os.path.join(final_output_dir, 'model_best_val_loss.pth.tar')
            torch.save({'model': model.module.state_dict()}, best_val_loss_model_path)
            logger.info("=> saved best validation loss model to {} ".format(best_val_loss_model_path))
        
        # 定期保存检查点
        if (epoch_i + 1) % 200 == 0:
            latest_checkpoint_path = os.path.join(final_output_dir, 'checkpoint_latest.pth.tar')
            torch.save({
                "epoch": epoch_i,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_train_loss': best_train_loss,
                'best_train_loss_epoch': best_train_loss_epoch,
                'best_val_loss': best_val_loss,
                'best_val_loss_epoch': best_val_loss_epoch,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'epoch_history': epoch_history
            }, latest_checkpoint_path)
            logger.info("=> saved latest checkpoint to {} ".format(latest_checkpoint_path))
    
    plot_loss_curves(logger, epoch_history, train_loss_history, val_loss_history, best_val_loss, best_val_loss_epoch, final_output_dir)
        
def test():
    gpus = [int(i) for i in config.GPUS.split(",")]
    device = torch.device('cuda')

    dataset_test = TrajectoryLoader(config=config, mode='test', inference=True)
    dataloader_test = DataLoader(dataset_test, batch_size=config.TRAIN.BATCH_SIZE, num_workers=6, pin_memory=True, shuffle=False)

    view_num = dataset_test.get_view_num()
    model = MultiViewTrajectoryEncoder(view_num, depth=config.DEPTH, trajectory_len=config.TRAJECTORY_LEN, mode=config.PROMPT_MODE)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    checkpoint_path = os.path.join(final_output_dir, config.CKP)
    ckp = torch.load(checkpoint_path)
    model.module.load_state_dict(ckp['model'])

    reid_model = None
    if config.REID:
        reid_model = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='weights/osnet_ain_ms_d_c.pth.tar',
            device='cuda')
        
    model.eval()
    with torch.no_grad():
        evaluate(config, model, dataloader_test, device, config.TEST.THRESH, reid_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument("--test", help="test mode", action='store_true', default=False)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    
    if args.test:
        logger, final_output_dir = create_logger(config, "test")
        filtered_config = {k: v for k, v in config.items() if k != 'TRAIN'}
        logger.info(pprint.pformat(filtered_config))
        test()
    else:
        logger, final_output_dir = create_logger(config, "train")
        filtered_config = {k: v for k, v in config.items() if k != 'TEST'}
        logger.info(pprint.pformat(filtered_config))
        train()