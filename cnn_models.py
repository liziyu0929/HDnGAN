# cnn_models.py
#
# CNN models including a 3D generator, a 2D discrimiator, and a hybrid GAN 
#
# (c) Ziyu Li, 2021

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Conv3D, concatenate, add, Multiply, BatchNormalization, Activation, ReLU, LeakyReLU,\
                         MaxPooling3D, UpSampling3D, Dense, Flatten, Dropout, Permute, concatenate
from keras.layers.core import Lambda
from keras.initializers import RandomNormal
from SpectralNormalizationKeras import DenseSN, ConvSN2D

num_slice = 64

def slice(x,index):
    return x[:,:,:,:,:index]

def conv3d_bn_relu(inputs, filter_num, bn_flag=False):
    if bn_flag:
        conv = Conv3D(filter_num, (3,3,3), padding='same', kernel_initializer='he_normal')(inputs)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)        
    else:
        conv = Conv3D(filter_num, (3,3,3), padding='same', kernel_initializer='he_normal')(inputs)
        conv = Activation('relu')(conv)
    return conv

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

def flatten_first2axes_operator_3d(x2d):
    shape = tf.shape(x2d) # get dynamic tensor shape
    x3d = tf.reshape(x2d, [shape[0]//(num_slice*3), num_slice*3, shape[1]])
    return x3d

def flatten_first2axes_shape_3d(x2d_shape):
    in_batch, in_filters = x2d_shape
    if (in_batch is None):
        output_shape = (None, num_slice * 3, in_filters)
    else:
        output_shape = (in_batch//(num_slice), num_slice * 3, in_filters)
    return output_shape

def generator_3d_model(num_ch = 1, filter_num=64, kinit_type='he_normal', tag='modified_unet3d'):
    
    inputs = Input((None, None, None, num_ch)) 
    bmask = Input((None, None, None, 1))
    loss_weights = Input((None, None, None, 1))
    
    p0 = Multiply()([inputs, bmask]) # presever ROI
    
    conv1 = conv3d_bn_relu(p0, filter_num)
    conv1 = conv3d_bn_relu(conv1, filter_num)
    
    conv2 = conv3d_bn_relu(conv1, filter_num)
    conv2 = conv3d_bn_relu(conv2, filter_num)

    conv3 = conv3d_bn_relu(conv2, filter_num)
    conv3 = conv3d_bn_relu(conv3, filter_num)
   
    conv4 = conv3d_bn_relu(conv3, filter_num)
    conv4 = conv3d_bn_relu(conv4, filter_num)

    conv5 = conv3d_bn_relu(conv4, filter_num)
    conv5 = conv3d_bn_relu(conv5, filter_num)

    merge6 = concatenate([conv4,conv5])
    conv6 = conv3d_bn_relu(merge6, filter_num)
    conv6 = conv3d_bn_relu(conv6, filter_num)
    
    merge7 = concatenate([conv3,conv6])
    conv7 = conv3d_bn_relu(merge7, filter_num)
    conv7 = conv3d_bn_relu(conv7, filter_num)

    merge8 = concatenate([conv2,conv7])
    conv8 = conv3d_bn_relu(merge8, filter_num)
    conv8 = conv3d_bn_relu(conv8, filter_num)

    merge9 = concatenate([conv1,conv8])
    conv9 = conv3d_bn_relu(merge9, filter_num)
    conv9 = conv3d_bn_relu(conv9, filter_num)
    
    residual = Conv3D(1, (3, 3, 3), padding='same',
                  activation=None, 
                  kernel_initializer='he_normal'
                  )(conv9)
        
     # add residual
    layer_name = tag + '_add_residual'
    recon = add([p0, residual], name=layer_name)
    recon = Multiply()([recon, bmask]) # presever ROI
    
    conv = concatenate([recon, loss_weights],axis=-1)
        
    model = Model(inputs=[inputs, bmask, loss_weights], outputs=conv)  
    
    return model


def discriminator_2d_model(img_size, num_ch=1, tag='discriminator_2d'):
    
    inputs = Input((img_size, img_size, num_ch)) # channel last
    initializer = RandomNormal(mean=0, stddev=0.02)
    
    df_dim = 64
    
    net_in = inputs
    net_h0 = ConvSN2D(df_dim, (3, 3), strides=(1, 1), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_in)
    net_h0 = LeakyReLU(alpha=0.2)(net_h0)
    net_h0 = ConvSN2D(df_dim, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h0)
    net_h0 = BatchNormalization()(net_h0)
    net_h0 = LeakyReLU(alpha=0.2)(net_h0)

    net_h1 = ConvSN2D(df_dim * 2, (3, 3), strides=(1, 1), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h0)
    net_h1 = BatchNormalization()(net_h1)
    net_h1 = LeakyReLU(alpha=0.2)(net_h1)
    net_h1 = ConvSN2D(df_dim * 2, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h1)
    net_h1 = BatchNormalization()(net_h1)
    net_h1 = LeakyReLU(alpha=0.2)(net_h1)
    
    net_h2 = ConvSN2D(df_dim * 4, (3, 3), strides=(1, 1), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h1)
    net_h2 = BatchNormalization()(net_h2)
    net_h2 = LeakyReLU(alpha=0.2)(net_h2)
    net_h2 = ConvSN2D(df_dim * 4, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h2)
    net_h2 = BatchNormalization()(net_h2)
    net_h2 = LeakyReLU(alpha=0.2)(net_h2)
    
    net_h3 = ConvSN2D(df_dim * 8, (3, 3), strides=(1, 1), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h2)
    net_h3 = BatchNormalization()(net_h3)
    net_h3 = LeakyReLU(alpha=0.2)(net_h3)
    net_h3 = ConvSN2D(df_dim * 8, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h3)
    net_h3 = BatchNormalization()(net_h3)
    net_h3 = LeakyReLU(alpha=0.2)(net_h3)
    
    net_ho = Flatten()(net_h3)
    net_ho = DenseSN(df_dim * 16, activation=None)(net_ho)
    net_ho = LeakyReLU(alpha=0.2)(net_ho)
    net_out = DenseSN(1, activation='sigmoid')(net_ho)
    
    model = Model(inputs = inputs, outputs = net_out)
    
    return model

def gan_hybrid_model(block_size, input_ch_g, input_ch_d, generator, discriminator,tag='hybrid GAN'):
    inputs = Input((block_size, block_size, block_size, input_ch_g))  
    bmask = Input((block_size, block_size, block_size, 1))
    loss_weights = Input((block_size, block_size, block_size, 1))
    
    generated_block = generator([inputs, bmask, loss_weights])
    
    generated_block_1 = Lambda(slice, arguments={'index':input_ch_d})(generated_block)
    generated_block_2 = Permute(dims=(2,1,3,4))(generated_block_1)
    generated_block_3 = Permute(dims=(3,1,2,4))(generated_block_1)
    generated_blocks = concatenate([generated_block_1, generated_block_2, generated_block_3],axis=0)
    generated_images = Lambda(squeeze_first2axes_operator, output_shape = squeeze_first2axes_shape)(generated_blocks)
    discriminator_outputs = discriminator(generated_images)
    discriminator_outputs = Lambda(flatten_first2axes_operator_3d, 
                                   output_shape = flatten_first2axes_shape_3d)(discriminator_outputs)
    
    model = Model(inputs=[inputs, bmask, loss_weights], outputs=[generated_block, discriminator_outputs])
    return model