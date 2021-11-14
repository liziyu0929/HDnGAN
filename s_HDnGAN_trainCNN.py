# s_HDnGAN_trainHDnGAN.py
#
#   A script for training HDnGAN for high-fidelity MRI denoising.
#
#   Source code:
#   https://github.com/liziyu0929/HDnGAN/blob/main/s_HDnGAN_trainCNN.py
#
#   Reference:
#   [1] Li, Z., Tian, Q., Ngamsombat, C., Cartmell, S., Conklin, J., Gonçalves Filho, A. L. M., ... & Huang, S. Y. (2021). 
#       High-fidelity fast volumetric brain MRI using synergistic wave-controlled aliasing in parallel imaging and a hybrid 
#       denoising generative adversarial network. bioRxiv: https://www.biorxiv.org/content/10.1101/2021.01.07.425779v2.abstract. 
#        (Submitted to Medical Physics).
#
#   [2] Li, Z. Oral Presentation. The 2021 Annual Scientific Meeting of ISMRM. Video link: 
#       https://cds.ismrm.org/protected/21MPresentations/videos//0390.htm. (Magna Cum Laude Merit Award).
#
# (c) Ziyu Li, Qiyuan Tian, 2021

# %% load moduals
import os
import glob
import scipy.io as sio
import numpy as np
import nibabel as nb
import tensorflow as tf
from matplotlib import pyplot as plt

from keras.optimizers import Adam

from cnn_models import generator_3d_model, discriminator_2d_model, gan_hybrid_model
import cnn_utils as utils

# for compatibility
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import GPUOptions

gpu_options = GPUOptions(per_process_gpu_memory_fraction=0.9)
config = ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# %% set up path and parameters
dpRoot = os.path.dirname(os.path.abspath('s_HDnGAN_trainCNN.py'))
os.chdir(dpRoot)

block_size = 64

# %% subjects
subjects = sorted(glob.glob(os.path.join(dpRoot, 'example_hcp_data', 'mwu*')))

# %% load data 
noisy_block_train = np.zeros(1)
noisy_block_test = np.zeros(1)

clean_block_train = np.zeros(1)
clean_block_test = np.zeros(1)

bmask_block_train = np.zeros(1)
bmask_block_test = np.zeros(1)

for ii in range(len(subjects)):
    sj = os.path.basename(subjects[ii])
    
    print(sj)
    dpSub = os.path.join(dpRoot, 'example_hcp_data', sj)
    
    fpNoisy = os.path.join(dpSub, sj + '_t1w_sim0.5.nii.gz')
    fpClean = os.path.join(dpSub, sj + '_t1w.nii.gz')
    fpBmask = os.path.join(dpSub, 'brainmask_fs_dil2.nii.gz')

    noisy_struct = nb.load(fpNoisy)
    clean_struct = nb.load(fpClean)
    bmask_struct = nb.load(fpBmask)

    noisy = np.squeeze(np.array(noisy_struct.dataobj))
    clean = np.squeeze(np.array(clean_struct.dataobj))
    bmask = np.squeeze(np.array(bmask_struct.dataobj))
    
    noisy = np.expand_dims(noisy, -1)
    clean = np.expand_dims(clean, -1)
    bmask = np.expand_dims(bmask, -1)
    
    # normalize
    clean_norm, noisy_norm = utils.normalize_image(clean, noisy, bmask)
    
    # get block
    [ind_block, ind_brain] = utils.block_ind(bmask, block_size, 1)
    noisy_block = utils.extract_block(noisy_norm, ind_block)
    clean_block = utils.extract_block(clean_norm, ind_block)
    bmask_block = utils.extract_block(bmask, ind_block)
    
    if np.mod(ii, 3) == 0:
        print('validation & evalution subject')
        # here for demonstration purpose we use the same subject for validation and evalution 
        if noisy_block_test.any():
            noisy_block_test = np.concatenate((noisy_block_test, noisy_block), axis=0)
            clean_block_test = np.concatenate((clean_block_test, clean_block), axis=0)
            bmask_block_test = np.concatenate((bmask_block_test, bmask_block), axis=0)
        else:
            noisy_block_test = noisy_block
            clean_block_test = clean_block
            bmask_block_test = bmask_block
            noisy_block_apply = noisy_block # for apply
            bmask_block_apply = bmask_block
            dpSub_apply = dpSub
            bmask_apply = bmask
            ind_apply = ind_block
    else:
        if noisy_block_train.any():
            noisy_block_train = np.concatenate((noisy_block_train, noisy_block), axis=0)
            clean_block_train = np.concatenate((clean_block_train, clean_block), axis=0)
            bmask_block_train = np.concatenate((bmask_block_train, bmask_block), axis=0)
        else:
            noisy_block_train = noisy_block
            clean_block_train = clean_block            
            bmask_block_train = bmask_block
            

# setup data
clean_output_train = np.concatenate((clean_block_train, bmask_block_train), axis=-1)
clean_output_test = np.concatenate((clean_block_test, bmask_block_test), axis=-1)

# %% set up models
input_ch_g = 1
input_ch_d = 1

model_generator = generator_3d_model(input_ch_g)
model_generator.summary()
model_discriminator = discriminator_2d_model(block_size, input_ch_d)
model_discriminator.summary()
model_discriminator.trainable = False
model_gan = gan_hybrid_model(block_size, input_ch_g, input_ch_d, model_generator, model_discriminator)
model_gan.summary()

# set up optimizer
opt_g = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
opt_d = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

# compile models
model_generator.compile(loss = utils.mean_squared_error_weighted, optimizer = opt_g)
model_discriminator.trainable = True
model_discriminator.compile(loss = 'binary_crossentropy', optimizer = opt_d)
model_discriminator.trainable = False
loss = [utils.mean_squared_error_weighted, 'binary_crossentropy']
loss_weights = [1, 1e-3]
model_gan.compile(optimizer = opt_g, loss = loss, loss_weights=loss_weights)
model_discriminator.trainable = True
    
# %% train
num_epochs = 20
l2_loss_train, l2_loss_test = [], []
gan_loss_train, gan_loss_test = [], []
d_loss_train, d_loss_test = [], []

fnCp = 'HDnGAN_lw1e-3'

total_train_num = clean_output_train.shape[0]
total_test_num = clean_output_test.shape[0]
print('Training on', total_train_num, 'blocks. Testing on', total_test_num, 'blocks.')
batch_size_train, batch_size_test = 1, 1

for ii in range(num_epochs):
    cnt_train, cnt_test = 0, 0
    
    # shuffle the images
    index_train = np.arange(total_train_num)
    np.random.shuffle(index_train)
    noisy_block_train = noisy_block_train[index_train,:,:,:,:]
    bmask_block_train = bmask_block_train[index_train,:,:,:,:]
    clean_output_train = clean_output_train[index_train,:,:,:,:]

    index_test = np.arange(total_test_num)
    np.random.shuffle(index_test)
    noisy_block_test = noisy_block_test[index_test,:,:,:,:]
    bmask_block_test = bmask_block_test[index_test,:,:,:,:]
    clean_output_test = clean_output_test[index_test,:,:,:,:]

    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('\n')
    print('Total epoch count:', ii + 1)
    gan_loss_train_batch, l2_loss_train_batch, d_loss_train_batch = [], [], []
    gan_loss_test_batch, l2_loss_test_batch, d_loss_test_batch = [], [], []
    while cnt_train + batch_size_train < total_train_num:
        if cnt_test + batch_size_test >= total_test_num:
            cnt_test = 0
        
        print('\n')
        print('Training blocks count:', cnt_train)
        
        # prepare training and testing batch
        train_batch_noisy = noisy_block_train[cnt_train:cnt_train+batch_size_train,:,:,:,:]
        train_batch_clean = clean_output_train[cnt_train:cnt_train+batch_size_train,:,:,:,:]
        train_batch_bmask = bmask_block_train[cnt_train:cnt_train+batch_size_train,:,:,:,:]
        train_batch_lmask = bmask_block_train[cnt_train:cnt_train+batch_size_train,:,:,:,:]
        
        test_batch_noisy = noisy_block_test[cnt_test:cnt_test+batch_size_test,:,:,:,:]
        test_batch_clean = clean_output_test[cnt_test:cnt_test+batch_size_test,:,:,:,:]
        test_batch_bmask = bmask_block_test[cnt_test:cnt_test+batch_size_test,:,:,:,:]
        test_batch_lmask = bmask_block_test[cnt_test:cnt_test+batch_size_test,:,:,:,:]
        
        # prepare labels and images for discriminator
        if ii == 0 and cnt_train == 0:
            generated_train = train_batch_noisy
            generated_test = test_batch_noisy
    
        else:
            generated_train = model_generator.predict([train_batch_noisy, train_batch_bmask, train_batch_lmask])[:,:,:,:,:input_ch_d]
            generated_test = model_generator.predict([test_batch_noisy, test_batch_bmask, test_batch_lmask])[:,:,:,:,:input_ch_d]
        
        generated_train_sag = generated_train
        generated_train_cor = np.transpose(generated_train,[0,2,1,3,4]) # generate images of different directions
        generated_train_axial = np.transpose(generated_train,[0,3,1,2,4])
        generated_test_sag = generated_test
        generated_test_cor = np.transpose(generated_test,[0,2,1,3,4])
        generated_test_axial = np.transpose(generated_test,[0,3,1,2,4])
        generated_train_all = np.concatenate((generated_train_sag, 
                                                generated_train_cor, generated_train_axial), axis=0)
        generated_test_all = np.concatenate((generated_test_sag, 
                                                generated_test_cor, generated_test_axial), axis=0)
        
        clean_train_sag = train_batch_clean
        clean_train_cor = np.transpose(train_batch_clean,[0,2,1,3,4]) 
        clean_train_axial = np.transpose(train_batch_clean,[0,3,1,2,4])
        clean_test_sag = test_batch_clean
        clean_test_cor = np.transpose(test_batch_clean,[0,2,1,3,4])
        clean_test_axial = np.transpose(test_batch_clean,[0,3,1,2,4])
        clean_train_all = np.concatenate((clean_train_sag, clean_train_cor, clean_train_axial), axis=0)
        clean_test_all = np.concatenate((clean_test_sag, clean_test_cor, clean_test_axial), axis=0)
        
        generated_train, generated_test = generated_train_all, generated_test_all
        clean_train, clean_test = clean_train_all, clean_test_all
        
        shape_train = np.shape(generated_train)
        shape_test = np.shape(generated_test)
        generated_train = np.reshape(generated_train,[shape_train[0]*shape_train[1],shape_train[2],shape_train[3],1])
        generated_test = np.reshape(generated_test,[shape_test[0]*shape_test[1],shape_test[2],shape_test[3],1])
        clean_train = np.reshape(clean_train,[shape_train[0]*shape_train[1],shape_train[2],shape_train[3],2])[:,:,:,:input_ch_d]
        clean_test = np.reshape(clean_test,[shape_test[0]*shape_test[1],shape_test[2],shape_test[3],2])[:,:,:,:input_ch_d]
        
        dtrain_input_image_pred = np.zeros(1)
        dtrain_input_image_clean = np.zeros(1)
        flag = 1
        for jj in range(np.shape(generated_train)[0]):
            if np.sum(np.abs(clean_train[jj]) > 0 ) > 400: # discard empty and extremely sparse slices
                flag = 0
                if dtrain_input_image_pred.any():
                    dtrain_input_image_pred = np.concatenate((dtrain_input_image_pred, np.expand_dims(generated_train[jj],0)), axis=0)
                    dtrain_input_image_clean = np.concatenate((dtrain_input_image_clean, np.expand_dims(clean_train[jj],0)), axis=0)
                else:
                    dtrain_input_image_pred = np.expand_dims(generated_train[jj],0)
                    dtrain_input_image_clean = np.expand_dims(clean_train[jj],0)
        if flag: # empty block
            print('empty training block!')
            cnt_train += batch_size_train
            cnt_test += batch_size_test
            print('Total epoch: ', ii + 1)
            continue
    
        dtest_input_image_pred = np.zeros(1)
        dtest_input_image_clean = np.zeros(1)    
        flag = 1
        for jj in range(np.shape(generated_test)[0]):
            if np.sum(np.abs(clean_test[jj]) > 0 ) > 400:
                flag = 0
                if dtest_input_image_pred.any():
                    dtest_input_image_pred = np.concatenate((dtest_input_image_pred, np.expand_dims(generated_test[jj],0)), axis=0)
                    dtest_input_image_clean = np.concatenate((dtest_input_image_clean, np.expand_dims(clean_test[jj],0)), axis=0)
                else:
                    dtest_input_image_pred = np.expand_dims(generated_test[jj],0)
                    dtest_input_image_clean = np.expand_dims(clean_test[jj],0)
        if flag: 
            print('empty testing block!')
            cnt_train += batch_size_train
            cnt_test += batch_size_test
            print('Total epoch: ', ii + 1)
            continue
        
        doutput_false_train_tag = np.zeros((1,np.shape(dtrain_input_image_pred)[0]))[0] 
        doutput_true_train_tag = np.ones((1,np.shape(dtrain_input_image_clean)[0]))[0] 
        doutput_false_test_tag = np.zeros((1,np.shape(dtest_input_image_pred)[0]))[0] 
        doutput_true_test_tag = np.ones((1,np.shape(dtest_input_image_clean)[0]))[0] 
        
        dtrain_input_image = np.concatenate((dtrain_input_image_pred, dtrain_input_image_clean), axis=0)
        dtrain_output_tag = np.concatenate((doutput_false_train_tag, doutput_true_train_tag), axis=0)
        dtest_input_image = np.concatenate((dtest_input_image_pred, dtest_input_image_clean), axis=0)
        dtest_output_tag = np.concatenate((doutput_false_test_tag, doutput_true_test_tag), axis=0)
    
        # train the discriminator
        print('----------------------------------------------------------------------')
        print('Training the discriminator')
        history1 = model_discriminator.fit(x = dtrain_input_image, 
                                        y = dtrain_output_tag,
                                        validation_data = (dtest_input_image,\
                                                                dtest_output_tag),
                                        batch_size = 10, 
                                        epochs = 3,  
                                        shuffle = True, 
                                        callbacks = None, 
                                        verbose = 2)
    
        model_discriminator.trainable = False
        gtrain_output_tag = np.ones((batch_size_train, block_size*3, 1)) 
        gtest_output_tag = np.ones((batch_size_test, block_size*3, 1)) 
            
        # train the GAN
        print('----------------------------------------------------------------------')
        print('Training the GAN')
        
        history2 = model_gan.fit(x = [train_batch_noisy, train_batch_bmask, train_batch_lmask], 
                                    y = [train_batch_clean, gtrain_output_tag],
                                    validation_data = ([test_batch_noisy, test_batch_bmask, test_batch_lmask], \
                                                        [test_batch_clean, gtest_output_tag]),
                                    batch_size = 1, 
                                    epochs = 1,  
                                    shuffle = True, 
                                    callbacks = None, 
                                    verbose = 2)
        
        l2_loss_train_batch.append(history2.history['model_1_loss'])
        gan_loss_train_batch.append(history2.history['lambda_3_loss'])
        d_loss_train_batch.append(history1.history['loss'])
        l2_loss_test_batch.append(history2.history['val_model_1_loss'])
        gan_loss_test_batch.append(history2.history['val_lambda_3_loss'])
        d_loss_test_batch.append(history1.history['val_loss'])
                                
        cnt_train += batch_size_train
        cnt_test += batch_size_test
        print('Total epoch: ', ii + 1)
        
    print('Discriminator loss: train:',np.mean(d_loss_train_batch),'test:', np.mean(d_loss_test_batch))
    print('GAN loss: train:',np.mean(gan_loss_train_batch),'test:', np.mean(gan_loss_test_batch))
    print('L2 loss: train:',np.mean(l2_loss_train_batch),'test:', np.mean(l2_loss_test_batch))
    d_loss_train.append(np.mean(d_loss_train_batch))
    d_loss_test.append(np.mean(d_loss_test_batch))
    gan_loss_train.append(np.mean(gan_loss_train_batch))
    gan_loss_test.append(np.mean(gan_loss_test_batch))
    l2_loss_train.append(np.mean(l2_loss_train_batch))
    l2_loss_test.append(np.mean(l2_loss_test_batch))
    
    
    fpCp1 = os.path.join(dpRoot, 'discriminator', fnCp + '_epoch' + str(ii + 1) + '.h5')
    fpCp2 = os.path.join(dpRoot, 'generator', fnCp + '_epoch' + str(ii + 1) + '.h5')
    fpLoss = os.path.join(dpRoot, 'loss', fnCp + '_loss.mat') 
    model_discriminator.save(fpCp1)
    model_generator.save(fpCp2)
    sio.savemat(fpLoss, {'l2_loss_train':l2_loss_train, 'l2_loss_test': l2_loss_test,
                        'gan_loss_train': gan_loss_train, 'gan_loss_test': gan_loss_test,
                        'd_loss_train': d_loss_train, 'd_loss_test': d_loss_test})
print('Training finished')

# %% apply 
print('Applying...')
clean_block_pred = []
for ii in range(len(noisy_block_apply)):
    noisy_block_tmp = np.expand_dims(noisy_block_apply[ii], 0)
    bmask_block_tmp = np.expand_dims(bmask_block_apply[ii], 0)
    clean_block_pred_tmp = np.squeeze(model_generator.predict([noisy_block_tmp, bmask_block_tmp, bmask_block_tmp])[:, :, :, :, 0])
    clean_block_pred.append(clean_block_pred_tmp)
clean_pred, vol_count = utils.block2brain(np.expand_dims(np.array(clean_block_pred), -1), ind_apply, bmask_apply)
fpPred = os.path.join(dpSub_apply, 'pred', fnCp +'_pred.nii.gz')
utils.save_nii(fpPred, clean_pred, fpClean)
print('Applying finished')