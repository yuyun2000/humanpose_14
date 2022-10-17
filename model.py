import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers

# （1）标准卷积模块
def conv_block(input_tensor, filters, alpha, kernel_size=(3, 3), strides=(1, 1)):
    # 超参数alpha控制卷积核个数
    filters = int(filters * alpha)

    # 卷积+批标准化+激活函数
    x = layers.Conv2D(filters, kernel_size,
                      strides=strides,  # 步长
                      padding='same',  # 0填充，卷积后特征图size不变
                      use_bias=False)(input_tensor)  # 有BN层就不需要计算偏置
    x = layers.BatchNormalization()(x)  # 批标准化
    x = layers.ReLU(6.0)(x)  # relu6激活函数
    return x  # 返回一次标准卷积后的结果

# （2）深度可分离卷积块
def depthwise_conv_block(input_tensor, point_filters, alpha, depth_multiplier, strides=(1, 1)):
    # 超参数alpha控制逐点卷积的卷积核个数
    point_filters = int(point_filters * alpha)

    # ① 深度卷积--输出特征图个数和输入特征图的通道数相同
    x = layers.DepthwiseConv2D(kernel_size=(3, 3),  # 卷积核size默认3*3
                               strides=strides,  # 步长
                               padding='same',  # strides=1时，卷积过程中特征图size不变
                               depth_multiplier=depth_multiplier,  # 超参数，控制卷积层中间输出特征图的长宽
                               use_bias=False)(input_tensor)  # 有BN层就不需要偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # relu6激活函数

    # ② 逐点卷积--1*1标准卷积
    x = layers.Conv2D(point_filters, kernel_size=(1, 1),  # 卷积核默认1*1
                      padding='same',  # 卷积过程中特征图size不变
                      strides=(1, 1),  # 步长为1，对特征图上每个像素点卷积
                      use_bias=False)(x)  # 有BN层，不需要偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # 激活函数

    return x  # 返回深度可分离卷积结果


def conv_block_withoutrelu(
        inputs,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1)
):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def mobileinvertedblock(inputs,inc,midc,outc,midkernelsize=(5,5)):
    x = conv_block(inputs,midc,1,kernel_size=(1,1))

    if inc >= outc:
        strides = (1,1)
    else:
        strides = (2,2)
    x = layers.DepthwiseConv2D(kernel_size=midkernelsize,
                               strides=strides,  # 步长
                               padding='same',  # strides=1时，卷积过程中特征图size不变
                               depth_multiplier=1,  # 超参数，控制卷积层中间输出特征图的长宽
                               use_bias=False)(x)  # 有BN层就不需要偏置
    x = layers.BatchNormalization()(x)  # 批标准化
    x = layers.ReLU(6.0)(x)  # relu6激活函数
    x = conv_block_withoutrelu(x,outc,kernel_size=(1,1))
    if inc == outc:
        return x+inputs
    else:
        return x
import numpy as np

def process_layer(image):
    np.random.seed(2022)#不同的顺序会影响最终的结果，设定随机的顺序减少这些影响
    mode = np.random.randint(3)
    if mode ==0:
        image = tf.image.random_brightness(image,max_delta=0.125)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif mode == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.125)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif mode == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=0.125)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif mode == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.125)

    return tf.clip_by_value(image,0.0,1.0)#把最终的结果限制在0-1的区间


def mobilenet_v1( input_shape,num_point=14):
    # 创建输入层
    inputs = layers.Input(shape=input_shape)
    inputs = process_layer(inputs)
    x = conv_block(inputs, 16, 1, strides=(2, 2))
    # x = conv_block(x, 12, 1)
    x2 = mobileinvertedblock(x,16,32,32) #x2 64*64*32

    x = mobileinvertedblock(x2, 32, 64, 64)
    x1 = depthwise_conv_block(x,64,1,1) #x1 32*32*64

    x = mobileinvertedblock(x1,64,80,128,midkernelsize=(3,3))
    x0 = mobileinvertedblock(x, 128, 96, 128) #x0 16*16*128

    x = mobileinvertedblock(x0, 128, 192, 160, midkernelsize=(3, 3))

    x00 = mobileinvertedblock(x,160,160,160,midkernelsize=(3, 3)) # 8 8 160


    x = mobileinvertedblock(x00,160,128,32,midkernelsize=(3, 3))

    x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(x)

    x = tf.concat((x,x0),3) #16 16 128+32
    x = conv_block(x, 32, 1)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(x)
    x = tf.concat((x,x1),3) #32 32 64+32
    x = conv_block(x, 32, 1)

    x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(x)
    x = tf.concat((x,x2),3)#64 64 32+32
    x = conv_block(x, 32, 1)


    x = conv_block(x,32,1,strides=(2,2))
    x = conv_block(x, 64, 1, strides=(2, 2))
    x = conv_block(x, 128, 1, strides=(2, 2))
    x = conv_block(x, 160, 1, strides=(2, 2))
    x = conv_block(x, 128, 1, strides=(2, 2))
    # x = layers.Reshape(target_shape=(2*2*128,))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(14*2)(x)

    # 构建模型
    model = Model(inputs, x)
    # 返回模型结构
    return model


from Flops import try_count_flops
if __name__ == '__main__':
    # 获得模型结构
    model = mobilenet_v1(input_shape=[256, 256,3])
    flops = try_count_flops(model)
    print(flops/1000000)
    # # 查看网络模型结构
    model.summary()
    # model.load_weights('./h5/pose-point.h5')

    # model.save("./mbtest.h5", save_format="h5")
    # print(model.layers[-3])

    # model = tf.keras.models.load_model("./mbtest.h5")
    # model.summary()