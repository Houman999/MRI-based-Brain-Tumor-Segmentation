# Loss Func and Metrics
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true + y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def categorical_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=[1, 2, 3])
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=[1, 2, 3])
    return K.mean(true_positives / (possible_positives + K.epsilon()), axis=0)

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)), axis=[1, 2, 3])
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)), axis=[1, 2, 3])
    return K.mean(true_negatives / (possible_negatives + K.epsilon()), axis=0)