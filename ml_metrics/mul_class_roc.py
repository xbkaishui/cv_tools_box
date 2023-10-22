import pytest
from loguru import logger
import torch
from torchmetrics import PrecisionRecallCurve, Recall, Accuracy
from torchmetrics.classification import ConfusionMatrix
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import recall_score
from scipy import interpolate
from torch import randn, randint
from torchmetrics import ROC, AUROC
import pandas as pd
from pathlib import Path

def save_plot_metric(px, py, save_dir=Path('.'), names=(), col_prefix='', avg_py=None):
    data_dict = {"px": px, "all": avg_py}
    for i, y in enumerate(py):
        label=f'{names[i]}'
        data_dict[label] = y
    df = pd.DataFrame(data_dict)
    df.fillna(0, inplace=True)
    df.to_csv(str(save_dir), index=False, float_format='%.4f')


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def test_multi_class_metric():
    from test_data import target, pred_label, pred_score
    
    logger.info("target shape {}", target.shape)
    logger.info("pred_score shape {}", pred_score.shape)
    logger.info("pred_label shape {}", pred_label.shape)

    num_classes = pred_score.shape[1]
    sample_size = len(target)
    pr_curve = PrecisionRecallCurve(task="multiclass", num_classes=num_classes)
    precisions, recalls, thresholds = pr_curve(pred_score, target)
    logger.info("precision len {}, shape {}", len(precisions), precisions[0].shape)
    
    linspace_size = max(500, sample_size)
    px = np.linspace(0, 1, linspace_size)
    precison_py = np.zeros((num_classes, linspace_size))
    recall_py =np.zeros((num_classes, linspace_size))
    pr_py = np.zeros((num_classes, linspace_size))
    f1_py = np.zeros((num_classes, linspace_size))
    
    for i in range(num_classes):
        precision = precisions[i]
        recall = recalls[i]
        threshold = thresholds[i]
        # 插值pr曲线处理
        interp_func = interpolate.interp1d(recall, precision, kind='linear', fill_value='extrapolate')
        smooth_precision = interp_func(px)
        pr_py[i] = smooth_precision
        
        # 计算 precsion recall 插值
        logger.info("class idx {} threshold.shape {}", i, threshold.shape)
        if threshold.shape == 1 or threshold.shape == torch.Size([]):
            recall_interp = np.zeros_like(px)
            precision_interp = np.zeros_like(px)
        else:
            recall_interp = np.interp(px, threshold.numpy(), recall.numpy()[:-1], left=1)
            precision_interp = np.interp(px, threshold.numpy(), precision.numpy()[:-1],left=0)
        precison_py[i] = precision_interp
        recall_py[i] = recall_interp
        f1_py[i] = 2 * precision_interp * recall_interp / (precision_interp + recall_interp + 1e-20)

    # save all metrics
    i = smooth(f1_py.mean(0), 0.1).argmax()  # max F1 index
    print(f'best f1 idx {i}')
    p, r, f1 = precison_py[:, i], recall_py[:, i], f1_py[:, i]
    best_confidence = px[i]
    best_precision = p.mean()
    best_recall = r.mean()
    best_f1 = f1.mean()
    logger.info("best confidence {}, best precision {}, best recall {}, best f1 {}", best_confidence, best_precision, best_recall, best_f1)
    accuracy = (pred_label == target).sum().item() / sample_size
    # calc fpr
    logger.info("accuracy {}", accuracy)
    # accuracy = Accuracy(task="multiclass", num_classes=num_classes, threshold=best_confidence, top_k=1)
    # acc_metric = accuracy(pred_score, target)
    # logger.info("acc metric {}", acc_metric)
    # calc confusion matrix
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes, threshold=best_confidence)
    confusion_matrix = confmat(pred_score, target).numpy()
    # logger.info("confusion_matrix result {}", confusion_matrix)
    
    # 初始化FPR和TNR列表
    fpr_list = []
    # fnr_list = []
    # tnr_list = []

    # 计算每个类别的FPR和TNR
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fn = np.sum(confusion_matrix[i, :]) - tp
        fp = np.sum(confusion_matrix[:, i]) - tp
        tn = np.sum(confusion_matrix) - (tp + fn + fp)
        
        fpr = fp / (fp + tn)
        # tnr = tn / (fp + tn)
        # fnr = fn / (tp + fn)
        
        fpr_list.append(fpr)
        # fnr_list.append(fnr)
        # tnr_list.append(tnr)

    # 计算平均FPR和TNR
    average_fpr = np.mean(fpr_list)
    # average_fnr = np.mean(fnr_list)
    # average_tnr = np.mean(tnr_list)
    logger.info("fpr {}, fnr {}", average_fpr, 1 - best_recall)
    
    # calc roc curve
    roc = ROC(task="multiclass", num_classes=num_classes)
    fprs, tprs, thresholds = roc(pred_score, target)
    roc_py = np.zeros((num_classes, linspace_size))
    for i in range(len(fprs)):
        fpr = fprs[i].numpy()
        tpr = tprs[i].numpy()
        interp_tpr = interpolate.interp1d(fpr, tpr, kind='linear')
        roc_y = interp_tpr(px)
        roc_py[i] = roc_y
    # auroc = AUROC(task="multiclass", num_classes=num_classes)
    # best_auc = auroc(pred_score, target)
    # logger.info("best auc {}", best_auc)
    save_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    names = [str(i) for i in range(num_classes)]
    save_plot_metric(px, roc_py, save_dir / "roc_curve.csv", names, 'roc', roc_py.mean(0))
    save_plot_metric(px, f1_py, save_dir / "f1_curve.csv", names, 'f1', f1_py.mean(0))
    save_plot_metric(px, pr_py, save_dir / "pr_curve.csv", names, 'pr', pr_py.mean(0))
    save_plot_metric(px, precison_py, save_dir / "p_curve.csv", names, 'r', precison_py.mean(0))
    save_plot_metric(px, recall_py, save_dir / "r_curve.csv", names, 'p', recall_py.mean(0))
    
    

def test_multi_class_pr_curve():
    num_classes = 5
    pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
                     [0.05, 0.75, 0.05, 0.05, 0.05],
                     [0.05, 0.05, 0.75, 0.05, 0.05],
                     [0.05, 0.05, 0.05, 0.75, 0.05]])
    target = torch.tensor([0, 1, 3, 2])
    count = 10000
    pred = randn(count, num_classes).softmax(dim=-1)
    target = randint(num_classes, (count,))
    pr_curve = PrecisionRecallCurve(task="multiclass", num_classes=num_classes)
    precisions, recalls, thresholds = pr_curve(pred, target)
    # logger.info("precision shape {}", precision.shape)
    logger.info("precison shape {}", precisions[0].shape)
    # plot first row
    logger.info("precision {}, recall {} thresholds {}", precisions, recalls, thresholds)
    
    precision = precisions[0]
    recall = recalls[0]
    interp_func = interpolate.interp1d(recall, precision, kind='linear', fill_value='extrapolate')
    smooth_recall = np.linspace(0, 1, max(200, count))  # 在0到1之间生成更多的Recall值
    smooth_precision = interp_func(smooth_recall)
    logger.info("smooth_recall shape {}", smooth_recall.shape)
    # 绘制平滑的Precision-Recall曲线
    plt.figure(figsize=(8, 6))
    plt.plot(smooth_recall, smooth_precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Smoothed Precision-Recall Curve')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # test_multi_class_pr_curve()
    test_multi_class_metric()