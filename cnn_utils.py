# -*- coding: utf-8 -*-
# Loss functions and other necessary tools
# Ziyu Li, Qiyuan Tian, 2020

import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Permute, concatenate
from keras.layers.core import Lambda
import nibabel as nb
import tensorflow.keras.backend as K

def block_ind(mask, sz_block=64, sz_pad=0):

    # find indices of smallest block that covers whole brain
    tmp = np.nonzero(mask)
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    
    xmin = np.min(xind); xmax = np.max(xind)
    ymin = np.min(yind); ymax = np.max(yind)
    zmin = np.min(zind); zmax = np.max(zind)
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]
    
    # calculate number of blocks along each dimension
    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1
    zlen = zmax - zmin + 1
    
    nx = int(np.ceil(xlen / sz_block)) + sz_pad
    ny = int(np.ceil(ylen / sz_block)) + sz_pad
    nz = int(np.ceil(zlen / sz_block)) + sz_pad
    
    # determine starting and ending indices of each block
    xstart = xmin
    ystart = ymin
    zstart = zmin
    
    xend = xmax - sz_block + 1
    yend = ymax - sz_block + 1
    zend = zmax - sz_block + 1
    
    xind_block = np.round(np.linspace(xstart, xend, nx))
    yind_block = np.round(np.linspace(ystart, yend, ny))
    zind_block = np.round(np.linspace(zstart, zend, nz))
    
    ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0]*zind_block.shape[0], 6])
    count = 0
    for ii in np.arange(0, xind_block.shape[0]):
        for jj in np.arange(0, yind_block.shape[0]):
            for kk in np.arange(0, zind_block.shape[0]):
                ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+sz_block-1, yind_block[jj], yind_block[jj]+sz_block-1, zind_block[kk], zind_block[kk]+sz_block-1])
                count = count + 1
    
    ind_block = ind_block.astype(int)
    
    return ind_block, ind_brain

def denormalize_image(imgall, imgnormall, mask):
    imgresall_denorm = np.zeros(imgall.shape)
    
    for jj in np.arange(imgall.shape[-1]):
        img = imgall[:, :, :, jj : jj + 1]
        imgres = imgnormall[:, :, :, jj : jj + 1]
        
        img_mean = np.mean(img[mask > 0.5])
        img_clean = np.std(img[mask > 0.5])
    
        imgres_norm = (imgres * img_clean + img_mean) * mask
        
        imgresall_denorm[:, :, :, jj : jj + 1] = imgres_norm
    return imgresall_denorm
    
def normalize_image(imgall, imgresall, mask):
    imgall_norm = np.zeros(imgall.shape)
    imgresall_norm = np.zeros(imgall.shape)
    
    for jj in np.arange(imgall.shape[-1]):
        img = imgall[:, :, :, jj : jj + 1]
        imgres = imgresall[:, :, :, jj : jj + 1]
        
        img_mean = np.mean(img[mask > 0.5])
        img_clean = np.std(img[mask > 0.5])
    
        img_norm = (img - img_mean) / img_clean * mask
        imgres_norm = (imgres - img_mean) / img_clean * mask
        
        imgall_norm[:, :, :, jj : jj + 1] = img_norm
        imgresall_norm[:, :, :, jj : jj + 1] = imgres_norm
    return imgall_norm, imgresall_norm
        
        
def extract_block(data, inds):
    xsz_block = inds[0, 1] - inds[0, 0] + 1
    ysz_block = inds[0, 3] - inds[0, 2] + 1
    zsz_block = inds[0, 5] - inds[0, 4] + 1
    ch_block = data.shape[-1]
    
    blocks = np.zeros((inds.shape[0], xsz_block, ysz_block, zsz_block, ch_block))
    
    for ii in np.arange(inds.shape[0]):
        inds_this = inds[ii, :]
        blocks[ii, :, :, :, :] = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :]
    
    return blocks

def slice(x,index):
    return x[:,:,:,:,:index]

def squeeze_first2axes_operator(x5d) :
    shape = tf.shape(x5d) # get dynamic tensor shape
    x4d = tf.reshape(x5d, [shape[0]*shape[1], shape[2], shape[3], shape[4]])
    return x4d

def squeeze_first2axes_shape(x5d_shape):
    in_batch, in_slice, in_rows, in_cols, in_filters = x5d_shape
    if (in_batch is None):
        output_shape = (None, in_rows, in_cols, in_filters)
    else:
        output_shape = (in_batch*in_slice, in_rows, in_cols, in_filters)
    return output_shape

def VGG_16_loss(y_true, y_pred):
    """ VGG loss for 2D image slice """
    
    loss_weights = y_true[:, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :-1] * loss_weights
    vgg_model = VGG16(include_top=False, weights='imagenet')
    pred_weighted = K.concatenate([y_pred_weighted, y_pred_weighted, y_pred_weighted])
    true_weighted = K.concatenate([y_true_weighted, y_true_weighted, y_true_weighted])
    feature_pred = vgg_model(pred_weighted)
    feature_true = vgg_model(true_weighted)
    return K.mean(K.square(feature_pred - feature_true)) 

def VGG_16_loss_3d(y_true, y_pred):
    """ VGG loss for 3D image volume """

    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    
    generated_block_1 = Lambda(slice, arguments={'index':1})(y_pred_weighted)
    generated_block_2 = Permute(dims=(2,1,3,4))(generated_block_1)
    generated_block_3 = Permute(dims=(3,1,2,4))(generated_block_1)
    generated_blocks = concatenate([generated_block_1, generated_block_2, generated_block_3],axis=0)
    generated_images = Lambda(squeeze_first2axes_operator, output_shape = squeeze_first2axes_shape)(generated_blocks)
        
    true_block_1 = Lambda(slice, arguments={'index':1})(y_true_weighted)
    true_block_2 = Permute(dims=(2,1,3,4))(true_block_1)
    true_block_3 = Permute(dims=(3,1,2,4))(true_block_1)
    true_blocks = concatenate([true_block_1, true_block_2, true_block_3],axis=0)
    true_images = Lambda(squeeze_first2axes_operator, output_shape = squeeze_first2axes_shape)(true_blocks)
        
    vgg_in_pred = K.concatenate([generated_images, generated_images, generated_images], axis = -1)
    vgg_in_true = K.concatenate([true_images, true_images, true_images], axis = -1)
        
    vgg_model = VGG16(include_top=False, weights='imagenet')
    
    feature_pred = vgg_model(vgg_in_pred)
    feature_true = vgg_model(vgg_in_true)
        
    return K.mean(K.square(feature_pred - feature_true))

def mean_squared_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights

    return K.mean(K.square(y_pred_weighted - y_true_weighted), axis=-1)

def mean_squared_error_weighted_2d(y_true, y_pred):
    loss_weights = y_true[:, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :-1] * loss_weights

    return K.mean(K.square(y_pred_weighted - y_true_weighted), axis=-1)

def mean_absolute_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    
    return K.mean(K.abs(y_pred_weighted - y_true_weighted), axis=-1)

def mean_absolute_error_weighted_2d(y_true, y_pred):
    loss_weights = y_true[:, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :-1] * loss_weights
    
    return K.mean(K.abs(y_pred_weighted - y_true_weighted), axis=-1)

def VGG_L2_2d(y_true, y_pred):
    """ Combined loss of VGG and MSE for 2D image slices"""
    loss_weight = [1, 8]
    vgg_loss = VGG_16_loss(y_true, y_pred)
    l2_loss = mean_squared_error_weighted_2d(y_true, y_pred)
    return 1/sum(loss_weight)*(loss_weight[0]*vgg_loss+loss_weight[1]*l2_loss)

def VGG_L1_2d(y_true, y_pred):
    """ Combined loss of VGG and MAE for 2D image slices """
    loss_weight = [1, 5]
    vgg_loss = VGG_16_loss(y_true, y_pred)
    l1_loss = mean_absolute_error_weighted_2d(y_true, y_pred)
    return 1/sum(loss_weight)*(loss_weight[0]*vgg_loss+loss_weight[1]*l1_loss)

def VGG_L2_3d(y_true, y_pred):
    """ Combined loss of VGG and MSE for 3D image slices """
    loss_weight = [1, 1]
    vgg_loss = VGG_16_loss_3d(y_true, y_pred)
    l2_loss = mean_squared_error_weighted(y_true, y_pred)
    return 1/sum(loss_weight) * (loss_weight[0]*vgg_loss + loss_weight[1]*l2_loss)

def extract_block(data, inds):
    """
    Extract 3D image blocks with overlapping from 3D image volumes.
    
    Args:
        data: 3D image volumes with shape (x, y, z, num_ch)
        inds: indices generated by MATLAB function volbind
    """
    xsz_block = inds[0, 1] - inds[0, 0] + 1
    ysz_block = inds[0, 3] - inds[0, 2] + 1
    zsz_block = inds[0, 5] - inds[0, 4] + 1
    ch_block = data.shape[-1]
    
    blocks = np.zeros((inds.shape[0], xsz_block, ysz_block, zsz_block, ch_block))
    
    for ii in np.arange(inds.shape[0]):
        inds_this = inds[ii, :]
        blocks[ii, :, :, :, :] = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :]
    
    return blocks

def block2brain(blocks, inds, mask):
    vol_brain = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    vol_count = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    
    for tt in np.arange(inds.shape[0]):
        inds_this = inds[tt, :]
        
        vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] = \
                vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] + blocks[tt, :, :, :, :]
        
        vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] = \
                vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] + 1.
    
    vol_count[vol_count < 0.5] = 1.
    vol_brain = vol_brain / vol_count 
    
    vol_brain = vol_brain * mask
    vol_count = vol_count * mask
    
    return vol_brain, vol_count 

def save_nii(fpNii, data, fpRef):
    
    new_header = header=nb.load(fpRef).header.copy()    
    new_img = nb.nifti1.Nifti1Image(data, None, header=new_header)    
    nb.save(new_img, fpNii)  