import os
import logging
import numpy as np
import h5py
import pandas as pd
from sklearn.metrics import roc_curve, auc,accuracy_score, f1_score,precision_score,recall_score,precision_recall_curve, average_precision_score
import tensorflow as tf
from tensorflow.keras.models import load_model
from DanQ_model import  DataGenerator

logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

testmat = h5py.File('data/test.mat', 'r')
logging.info('test data loaded')
test_x = np.array(testmat['testxdata'])
test_y = np.array(testmat['testdata'])

model = tf.keras.models.load_model('weights/DanQ_bestmodel.hdf5')
logging.info('model loaded')

# Prediction
y_pred = model.predict(test_x)
logging.info('model predict')

# transfrom y_pred to binary form
y_pred_binary = (y_pred > 0.5).astype(int)


# calculate roc and auc
fpr, tpr, _ = roc_curve(test_y.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

from scipy.stats import pearsonr, spearmanr
# calculate pr_auc and average precision
precision, recall, _ = precision_recall_curve(test_y.ravel(), y_pred.ravel())
average_precision = average_precision_score(test_y, y_pred, average="micro")



import matplotlib.pyplot as plt
# ROC plot
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Micro-average)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# PRAUC plot
plt.figure()
plt.plot(recall, precision, label=f'Precision-recall curve (area = {average_precision:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Micro-average)')
plt.legend(loc="lower left")
plt.savefig('prauc_curve.png')
plt.close()

# Calculate Pearson correlation coefficient and Spearman correlation coefficient
average_precision = precision_score(test_y, y_pred_binary, average='micro', zero_division=0)
average_recall = recall_score(test_y, y_pred_binary, average='micro', zero_division=0)
average_f1 = f1_score(test_y, y_pred_binary, average='micro', zero_division=0)
pearson_corr, _ = pearsonr(test_y.ravel(), y_pred.ravel())
spearman_corr, _ = spearmanr(test_y.ravel(), y_pred.ravel())
logging.info(f"Average precision: {average_precision:.4f}")
logging.info(f"Average recall: {average_recall:.4f}")
logging.info(f"Average F1: {average_f1:.4f}")
logging.info(f"Pearson correlation coefficient: {pearson_corr:.4f}")
logging.info(f"Spearman correlation coefficient: {spearman_corr:.4f}")

