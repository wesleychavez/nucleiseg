# train.py, along with config.py trains U-Net on images of segmented nuclei
# for kaggle's 2018 Data Science Bowl.  It performs a grid search on batch 
# sizes and optimizers, and k-fold cross-validation on the dataset.  Metrics
# plots are saved for each hyperparameter combination.
#
# Wesley Chavez, 01-01-2019
import os
import sys
import random
import config
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
import itertools
from sklearn.model_selection import KFold
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def mean_iou(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value

# Build U-Net model
def create_model(img_height,img_width,img_channels,opt):
    inputs = Input((img_height,img_width,img_channels))
    s = Lambda(lambda x: x / 255) (inputs)
    
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()
    return model

def main():
    train_path = config.train_path
    img_width = config.img_width
    img_height = config.img_height
    img_channels = config.img_channels
    seed = config.randomseed
    random.seed = seed
    np.random.seed = seed
    train_ids = next(os.walk(train_path))[1]
    
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), img_height, img_width, img_channels), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), img_height, img_width, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = train_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((img_height, img_width, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant', 
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
    
    print('Done!')
    
    # Hyperparameters to optimize
    param_grid = [config.optimizer, config.batch_size]
    # Hyperparameter grid
    combos = list(itertools.product(*param_grid))

    # Metrics arrays, per epoch, per k fold
    mean_ious = np.zeros((len(combos), config.k, config.epochs))
    losses = np.zeros((len(combos), config.k, config.epochs))
    val_mean_ious = np.zeros((len(combos), config.k, config.epochs))
    val_losses = np.zeros((len(combos), config.k, config.epochs))

    # Loop over k folds of training/testing data
    kf = KFold(n_splits=config.k, shuffle=True, random_state=seed)
    k_iter = -1    
    for train_index, val_index in kf.split(np.zeros(X_train.shape[0])):
        k_iter = k_iter + 1
        x_train, x_val = X_train[train_index,:,:,:], X_train[val_index,:,:,:]
        y_train, y_val = Y_train[train_index,:,:,:], Y_train[val_index,:,:,:]

        # Grid Search, plotting cross-validation scores by epoch
        for i in range(len(combos)):
            model = create_model(img_height=img_height, img_width=img_width, img_channels=img_channels, opt=combos[i][0])

            history = model.fit(x_train, y_train, batch_size=combos[i][1], epochs=config.epochs, validation_data=(x_val, y_val))

            mean_ious[i,k_iter,:] = history.history['mean_iou']
            losses[i,k_iter,:] = history.history['loss']
            val_mean_ious[i,k_iter,:] = history.history['val_mean_iou']
            val_losses[i,k_iter,:] = history.history['val_loss']

    # Mean and std across k folds
    mean_iou = np.mean(mean_ious,axis=1)
    mean_val_iou = np.mean(val_mean_ious,axis=1)
    mean_loss = np.mean(losses,axis=1)
    mean_val_loss = np.mean(val_losses,axis=1)
    std_iou = np.std(mean_ious,axis=1)
    std_val_iou = np.std(val_mean_ious,axis=1)
    std_loss = np.std(losses,axis=1)
    std_val_loss = np.std(val_losses,axis=1)

    # Write these mean and stds to file organized by hyperparameter combination
    f = open("Metrics.txt","a")
    for i in range(len(combos)):
        f.write('-------------------------')
        f.write('\n')
        f.write(str(combos[i]))
        f.write('\n')
        f.write('-------------------------')
        f.write('\n')
        f.write('mean iou')
        f.write('\n')
        np.savetxt(f, mean_iou[i], delimiter=",", fmt='%.4f')
        f.write('std')
        f.write('\n')
        np.savetxt(f, std_iou[i], delimiter=",", fmt='%.4f')
        f.write('mean validation iou')
        f.write('\n')
        np.savetxt(f, mean_val_iou[i], delimiter=",", fmt='%.4f')
        f.write('std')
        f.write('\n')
        np.savetxt(f, std_val_iou[i], delimiter=",", fmt='%.4f')
        f.write('loss')
        f.write('\n')
        np.savetxt(f, mean_loss[i], delimiter=",", fmt='%.4f')
        f.write('std')
        f.write('\n')
        np.savetxt(f, std_loss[i], delimiter=",", fmt='%.4f')
        f.write('validation loss')
        f.write('\n')
        np.savetxt(f, mean_val_loss[i], delimiter=",", fmt='%.4f')
        f.write('std')
        f.write('\n')
        np.savetxt(f, std_val_loss[i], delimiter=",", fmt='%.4f')
    f.close()

    # Plot accuracies/losses for each hyperparameter combination
    # Each data point plotted is an average across the k folds.
    for i in range(len(combos)):
        fig, ax = plt.subplots(1,2)
        ax[0].errorbar(range(len(mean_iou[i])), mean_iou[i], yerr=std_iou[i])
        ax[0].errorbar(range(len(mean_val_iou[i])), mean_val_iou[i], yerr=std_val_iou[i])
        ax[0].set_title('Model IOU')
        ax[0].set_ylabel('IOU')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(['Train', 'Val'], loc='upper left')

        ax[1].errorbar(range(len(mean_loss[i])), mean_loss[i], yerr=std_loss[i])
        ax[1].errorbar(range(len(mean_val_loss[i])), mean_val_loss[i], yerr=std_val_loss[i])
        ax[1].set_title('Model Loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(['Train', 'Val'], loc='upper left')
        out_name = str(combos[i])
        out_name = out_name.replace(" ", "")
        out_name = out_name.replace("'", "")
        out_name = out_name.replace("(", "")
        out_name = out_name.replace(")", "")
        out_name = out_name.replace(",", "_")
        fig.savefig(out_name + '_iouandloss.png')

    # Highest validation iou, at any epoch, for any data fold
    max_val_iou_bymodel = []
    for i in range(len(combos)):
        max_val_iou_bymodel.append(np.max(val_mean_ious[i,:,:]))
    #mean_val_iou_bymodel = np.mean(val_mean_ious[:,:,-1],axis=1)
    print('Model with highest mean validation iou:')
    print (combos[np.argmax(max_val_iou_bymodel)])
    print (np.max(max_val_iou_bymodel))
    print ('------------------------------')
    for i in range(len(max_val_iou_bymodel)):
        print(combos[i])
        print(max_val_iou_bymodel[i])

if __name__  == '__main__':
    main()
