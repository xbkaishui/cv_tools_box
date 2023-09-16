import torch
import torchmetrics as tm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

if __name__ == '__main__':

    # 创建一个示例的真实标签和模型预测结果
    true_labels = torch.Tensor([0, 1, 1, 0, 0, 1, 0, 1])
    predicted_probabilities = torch.Tensor([0.2, 0.8, 0.7, 0.3, 0.4, 0.9, 0.1, 0.6])

    # 使用torchmetrics计算Precision-Recall Curve
    precision_recall_curve = tm.PrecisionRecallCurve()
    precision, recall, threshold = precision_recall_curve(predicted_probabilities, true_labels)

    # 获取计算得到的精确度和召回率
    # precision, recall, _ = precision_recall_curve.precision, precision_recall_curve.recall

    # 对曲线进行插值处理
    interp_precision = interp1d(recall, precision, kind='linear')

    # 生成新的召回率值以获得更平滑的曲线
    smooth_recall = np.linspace(0, 1, 100)
    smooth_precision = interp_precision(smooth_recall)

    # 绘制平滑处理后的Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(smooth_recall, smooth_precision, color='b', lw=2, label='Smoothed Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Smoothed Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
