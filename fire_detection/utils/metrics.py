import numpy as np
import tensorflow as tf
from scipy import stats

def iou_metric(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def precision_metric(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    all_positives = tf.reduce_sum(y_pred)
    return (true_positives + 1e-7) / (all_positives + 1e-7)

def recall_metric(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    actual_positives = tf.reduce_sum(y_true)
    return (true_positives + 1e-7) / (actual_positives + 1e-7)

def f1_score_metric(y_true, y_pred, threshold=0.5):
    precision = precision_metric(y_true, y_pred, threshold)
    recall = recall_metric(y_true, y_pred, threshold)
    return 2 * (precision * recall) / (precision + recall + 1e-7)

def calculate_map50(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    true_positives = tf.reduce_sum(y_true * y_pred)
    all_positives = tf.reduce_sum(y_pred)
    precision = (true_positives + 1e-7) / (all_positives + 1e-7)
    ap50 = tf.where(iou >= 0.5, precision, 0.0)
    return tf.reduce_mean(ap50)

def calculate_statistical_significance(true_labels, preds_model1, preds_model2, n_bootstraps=1000):
    def compute_f1(y_true, y_pred):
        tp = np.sum(y_true * y_pred)
        fp = np.sum(y_pred) - tp
        fn = np.sum(y_true) - tp
        return 2*tp / (2*tp + fp + fn + 1e-7)
    
    bootstrapped_diffs = []
    n_samples = len(true_labels)
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        f1_1 = compute_f1(true_labels[indices], preds_model1[indices])
        f1_2 = compute_f1(true_labels[indices], preds_model2[indices])
        bootstrapped_diffs.append(f1_1 - f1_2)
    
    ci_low, ci_high = np.percentile(bootstrapped_diffs, [2.5, 97.5])
    _, p_value = stats.ttest_rel(preds_model1, preds_model2)
    
    return {
        'confidence_interval': (float(ci_low), float(ci_high)),
        'p_value': float(p_value),
        'mean_difference': np.mean(bootstrapped_diffs)
    }

class mAP5095(tf.keras.metrics.Metric):
    def __init__(self, name='MeanIoU5095', **kwargs):
        super(mAP5095, self).__init__(name=name, **kwargs)
        self.ious = self.add_weight(name='ious', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        thresholds = np.arange(0.5, 1.0, 0.05)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)

        batch_ious = []
        for t in thresholds:
            intersection = tf.reduce_sum(tf.cast((y_true * y_pred) > 0, tf.float32))
            union = tf.reduce_sum(tf.cast((y_true + y_pred) > 0, tf.float32))
            iou = (intersection + 1e-7) / (union + 1e-7)
            batch_ious.append(iou)

        self.ious.assign_add(tf.reduce_mean(batch_ious))
        self.count.assign_add(1.0)

    def result(self):
        return self.ious / self.count

    def reset_state(self):
        self.ious.assign(0.0)
        self.count.assign(0.0)