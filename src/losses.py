import tensorflow as tf

def detection_metric(y_true, y_pred):
    """BCE for Eddy Mask (Channel 0)."""
    det_true = y_true[..., 0:1]
    det_pred = y_pred[..., 0:1]
    return tf.keras.losses.binary_crossentropy(det_true, det_pred, from_logits=True)

def temp_metric(y_true, y_pred):
    """MSE for Temperature Anomaly (Channel 1)."""
    temp_true = y_true[..., 1:2]
    temp_pred = y_pred[..., 1:2]
    return tf.keras.losses.mean_squared_error(temp_true, temp_pred)

def boundary_sst_coupling_loss(y_true, y_pred, branch_align=False):
    """Custom loss to force spatial agreement between mask and SST borders."""
    mask_true = y_true[..., 0:1]
    sst_true = y_true[..., 1:2]
    mask_pred_prob = tf.math.sigmoid(y_pred[..., 0:1])
    sst_pred = y_pred[..., 1:2]

    ksize = 5
    inv_mask_true = 1.0 - mask_true
    eroded_true = 1.0 - tf.nn.max_pool2d(inv_mask_true, ksize=ksize, strides=1, padding='SAME')
    boundary_zone_true = mask_true - eroded_true

    inv_mask_pred = 1.0 - mask_pred_prob
    eroded_pred = 1.0 - tf.nn.max_pool2d(inv_mask_pred, ksize=ksize, strides=1, padding='SAME')
    boundary_zone_pred = mask_pred_prob - eroded_pred

    target_product = boundary_zone_true * sst_true
    predicted_product = boundary_zone_pred * sst_pred
    
    product_mse = tf.reduce_mean(tf.square(target_product - predicted_product))
    if branch_align:
        border_alignment = tf.reduce_mean(tf.square(boundary_zone_true - boundary_zone_pred))
        product_mse += 0.1 * border_alignment
    return product_mse

def gradient_loss(y_true, y_pred):
    """Ensures sharp temperature boundaries."""
    dy_true, dx_true = tf.image.image_gradients(y_true[..., 1:2])
    dy_pred, dx_pred = tf.image.image_gradients(y_pred[..., 1:2])
    return tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true))

def combined_eddy_loss(y_true, y_pred):    
    bce = detection_metric(y_true, y_pred)
    mse = temp_metric(y_true, y_pred)
    coupling = boundary_sst_coupling_loss(y_true, y_pred, branch_align=True)
    grad = gradient_loss(y_true, y_pred)
    return bce + (1000 * mse) + (1000 * coupling) + grad