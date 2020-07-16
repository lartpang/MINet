# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 上午9:54
# @Author  : Lart Pang
# @FileName: metric.py
# @Project : MINet
# @GitHub  : https://github.com/lartpang

import numpy as np


def cal_pr_mae_meanf(prediction, gt):
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    # 确保图片和真值相同 ##################################################
    # if prediction.shape != gt.shape:
    #     prediction = Image.fromarray(prediction).convert('L')
    #     gt_temp = Image.fromarray(gt).convert('L')
    #     prediction = prediction.resize(gt_temp.size)
    #     prediction = np.array(prediction)

    # 获得需要的预测图和二值真值 ###########################################
    if prediction.max() == prediction.min():
        prediction = prediction / 255
    else:
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    hard_gt = np.zeros_like(gt)
    hard_gt[gt > 128] = 1

    # MAE ##################################################################
    mae = np.mean(np.abs(prediction - hard_gt))

    # MeanF ################################################################
    threshold_fm = 2 * prediction.mean()
    if threshold_fm > 1:
        threshold_fm = 1
    binary = np.zeros_like(prediction)
    binary[prediction >= threshold_fm] = 1
    tp = (binary * hard_gt).sum()
    if tp == 0:
        meanf = 0
    else:
        pre = tp / binary.sum()
        rec = tp / hard_gt.sum()
        meanf = 1.3 * pre * rec / (0.3 * pre + rec)

    # PR curve #############################################################
    t = np.sum(hard_gt)
    precision, recall = [], []
    for threshold in range(256):
        threshold = threshold / 255.0
        hard_prediction = np.zeros_like(prediction)
        hard_prediction[prediction >= threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)
        if tp == 0:
            precision.append(0)
            recall.append(0)
        else:
            precision.append(tp / p)
            recall.append(tp / t)

    return precision, recall, mae, meanf


# MaxF #############################################################
def cal_maxf(ps, rs):
    assert len(ps) == 256
    assert len(rs) == 256
    maxf = []
    for p, r in zip(ps, rs):
        if p == 0 or r == 0:
            maxf.append(0)
        else:
            maxf.append(1.3 * p * r / (0.3 * p + r))

    return max(maxf)
