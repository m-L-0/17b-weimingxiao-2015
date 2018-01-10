import tensorflow as tf

# filename_queue = tf.train.string_input_producer('/home/r/captcha/data/captcha/labels/labels.')
import os
import csv
import skimage.io as io
from PIL import Image
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
for root, sub_folders, files in os.walk('./data/captcha/images/'):
        for name in files:
            infile = os.path.join(root, name)
            outfile = os.path.join(root, name)
            im = Image.open(infile)
            (x, y) = im.size  # read image size
            x_s = 56  # define standard width
            y_s = 40  # calc height based on standard width
            out = im.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
            out = out.convert('1')
            out.save(outfile)
csvfile = csv.reader(open('/home/r/captcha/data/captcha/labels/labels.csv', 'r'))
images = []
labels = []
for line in csvfile:
    a='./'+line[0]
    images.append(a)
    labels.append(line[1])
# print(images[0])
# print(labels[0])
temp = np.array([images, labels])
temp = temp.transpose()
np.random.shuffle(temp)
print(temp[0])
myone = []
mytwo = []
mythree = []
myfour = []
# tra_image_list = list(temp[:32000, 0])
# tra_label_list = list(temp[:32000, 1])
             
# val_image_list = list(temp[32000:36000, 0])
# val_label_list = list(temp[32000:36000, 1])
# test_image_list = list(temp[36000:, 0])
# test_label_list = list(temp[36000:, 1])
# tra_label_list = [int(float(i)) for i in tra_label_list]
# val_label_list = [int(float(i)) for i in val_label_list]
# test_label_list = [int(float(i)) for i in test_label_list]
def test(temp, myone, mytwo, mythree, myfour):
    a=0
    b=0
    c=0
    d=0
    for i in range(40000) :
        if (int(temp[i,1]) >9 and int(temp[i,1])<100):
            a=a+1
            mytwo.append(temp[i])
        elif (int(temp[i,1])<10) :
            b=b+1
            myone.append(temp[i])
        elif (int(temp[i,1])>99 and int(temp[i,1])<1000):
            c=c+1
            mythree.append(temp[i])
        else:
            d=d+1  
            myfour.append(temp[i])
    print('一位验证码有%d个' % b)
    print('两位验证码有%d个' % a)
    print('三位验证码有%d个' % c)
    print('四位验证码有%d个' % d)
def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def convert_to_tfrecord(images, labels, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''
    
    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)
    
    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], n_samples))
    
    
    
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images[i])  # type(image) must be array!
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                            'label': int64_feature(label),
                            'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' % e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')


# save_dir = './data'
# convert_to_tfrecord(tra_image_list, tra_label_list, save_dir, 'train')
# convert_to_tfrecord(val_image_list, val_label_list, save_dir, 'validation')
# convert_to_tfrecord(test_image_list, test_label_list, save_dir, 'test')
test(temp, myone, mytwo, mythree, myfour)

# tra_image_list = list(myone[:8000, 0]+mytwo[:8000, 0]+mythree[:8000, 0]+myfour[:8000, 0])

# tra_label_list = []
myone = np.array(myone)
np.random.shuffle(myone)
mytwo = np.array(mytwo)
mythree = np.array(mythree)
myfour = np.array(myfour)

tra_image_list1 = list(myone[:1100, 0])+list(mytwo[:900, 0])+list(mythree[:1100, 0])+list(myfour[:1000, 0])
tra_label_list1 = list(myone[:1100, 1])+list(mytwo[:900, 1])+list(mythree[:1100, 1])+list(myfour[:1000, 1])
tra_image_list2 = list(myone[1100:2200, 0])+list(mytwo[900:1800, 0])+list(mythree[1100:2200, 0])+list(myfour[1000:2000, 0])
tra_label_list2 = list(myone[1100:2200, 1])+list(mytwo[900:1800, 1])+list(mythree[1100:2200, 1])+list(myfour[1000:2000, 1])
tra_image_list3 = list(myone[2200:3300, 0])+list(mytwo[1800:2700, 0])+list(mythree[2200:3300, 0])+list(myfour[2000:3000, 0])
tra_label_list3 = list(myone[2200:3300, 1])+list(mytwo[1800:2700, 1])+list(mythree[2200:3300, 1])+list(myfour[2000:3000, 1])
tra_image_list4 = list(myone[3300:4400, 0])+list(mytwo[2700:3600, 0])+list(mythree[3300:4400, 0])+list(myfour[3000:4000, 0])
tra_label_list4 = list(myone[3300:4400, 1])+list(mytwo[2700:3600, 1])+list(mythree[3300:4400, 1])+list(myfour[3000:4000, 1])
tra_image_list5 = list(myone[4400:5500, 0])+list(mytwo[3600:4500, 0])+list(mythree[4400:5500, 0])+list(myfour[4000:5000, 0])
tra_label_list5 = list(myone[4400:5500, 1])+list(mytwo[3600:4500, 1])+list(mythree[4400:5500, 1])+list(myfour[4000:5000, 1])
tra_image_list6 = list(myone[5500:6600, 0])+list(mytwo[4500:5400, 0])+list(mythree[5500:6600, 0])+list(myfour[5000:6000, 0])
tra_label_list6 = list(myone[5500:6600, 1])+list(mytwo[4500:5400, 1])+list(mythree[5500:6600, 1])+list(myfour[5000:6000, 1])
val_image_list = list(myone[6600:7700, 0])+list(mytwo[5400:5600, 0])+list(mythree[6600:7700, 0])+list(myfour[6000:7000, 0])
val_label_list = list(myone[6600:7700, 1])+list(mytwo[5400:5600, 1])+list(mythree[6600:7700, 1])+list(myfour[6000:7000, 1])
test_image_list = list(myone[7700:, 0])+list(mytwo[5600:, 0])+list(mythree[7700:, 0])+list(myfour[7000:, 0])
test_label_list = list(myone[7700:, 1])+list(mytwo[5600:, 1])+list(mythree[7700:, 1])+list(myfour[7000:, 1])
# val_image_list = list(myone[1250:1407, 0])+list(mytwo[1250:1407, 0])+list(mythree[1250:1407, 0])+list(myfour[1250:1407, 0])
# val_label_list = list(myone[1250:1407, 1])+list(mytwo[1250:1407, 1])+list(mythree[1250:1407, 1])+list(myfour[1250:1407, 1])
# test_image_list = list(myone[1407:1563, 0])+list(mytwo[1407:1563, 0])+list(mythree[1407:1563, 0])+list(myfour[1407:1563, 0])
# test_label_list = list(myone[1407:1563, 1])+list(mytwo[1407:1563, 1])+list(mythree[1407:1563, 1])+list(myfour[1407:1563, 1])
# tra_label_list = [int(float(i)) for i in tra_label_list]
# val_label_list = [int(float(i)) for i in val_label_list]
# test_label_list = [int(float(i)) for i in test_label_list]
save_dir = './data'
convert_to_tfrecord(tra_image_list1, tra_label_list1, save_dir, 'train1')
convert_to_tfrecord(tra_image_list2, tra_label_list2, save_dir, 'train2')
convert_to_tfrecord(tra_image_list3, tra_label_list3, save_dir, 'train3')
convert_to_tfrecord(tra_image_list4, tra_label_list4, save_dir, 'train4')
convert_to_tfrecord(tra_image_list5, tra_label_list5, save_dir, 'train5')
convert_to_tfrecord(tra_image_list6, tra_label_list6, save_dir, 'train6')
convert_to_tfrecord(val_image_list, val_label_list, save_dir, 'validation')
convert_to_tfrecord(test_image_list, test_label_list, save_dir, 'test')