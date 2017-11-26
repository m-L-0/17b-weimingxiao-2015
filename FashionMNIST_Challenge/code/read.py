import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
def read_and_decode(tfrecords_file, batch_size=25):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file
    batch_size = batch_size
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    
    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.
    image = tf.reshape(image, [28, 28])
    image = tf.cast(image,tf.float32)*(1./255)
    label = tf.cast(img_features['label'], tf.int32)    
    image_batch, label_batch = tf.train.batch([image,label], batch_size = batch_size, capacity=2000)
    with tf.Session() as sess:
        i = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        try:
            while not coord.should_stop() and i < 1:
                # just plot one batch size 
                
                image, label = sess.run([image_batch, label_batch])
                #plot_images(image, label)
                # image = image.reshape(784)
                i += 1
                
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
    coord.join(threads)
    return image, label
# tfrecords_file = '/home/r/fashion-mnist/test.tfrecords'
# [image, label] = read_and_decode(tfrecords_file,5000)
# s,d,f=image.shape
# b=np.empty((s,d*f))
# for i in range(s):
#     c=image[i]
#     b[i]=c.reshape(d*f)
# print(b.shape)