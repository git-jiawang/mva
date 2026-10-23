import torch

def triplet_loss(fea1, fea2, labels1, labels2):
    anchor_idxes = []
    pos_idxes = []
    neg_idxes = []
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels2):
            if label1 == label2:
                for k, neg_label2 in enumerate(labels2):
                    if j == k:
                        continue
                    anchor_idxes.append(i)
                    pos_idxes.append(j)
                    neg_idxes.append(k)
    if len(anchor_idxes) == 0:
        return -1
    anchor_fea = fea1[anchor_idxes]
    pos_fea = fea2[pos_idxes]
    neg_fea = fea2[neg_idxes]
    pos_dis = torch.sqrt((anchor_fea - pos_fea).pow(2).sum(-1))
    neg_dis = torch.sqrt((anchor_fea - neg_fea).pow(2).sum(-1))
    M = 1.0
    loss = torch.nn.functional.relu(pos_dis - neg_dis + M).mean()
    
    return loss