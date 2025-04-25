import tensorflow as tf

def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    pos_loss = -alpha * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred) * y_true
    neg_loss = -(1 - alpha) * tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred) * (1.0 - y_true)
    loss = tf.reduce_mean(pos_loss + neg_loss)
    return loss

def combined_loss(y_true, y_pred, alpha=0.5):
    dl = dice_loss(y_true, y_pred)
    fl = focal_loss(y_true, y_pred)
    return alpha * dl + (1 - alpha) * fl