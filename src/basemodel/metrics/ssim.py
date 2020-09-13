def ssim_metric(y_true, y_pred):
    import tensorflow as tf
    return tf.image.ssim(y_true, y_pred, max_val=1.)