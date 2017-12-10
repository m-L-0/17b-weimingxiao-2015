import numpy as np
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import os
import read
import random
# mnist = input_data.read_data_sets('data/fashion', one_hot=True)
FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_string('my_list', '/home/r/mycar/cnn1',"""存放模型的目录""")

tf.app.flags.DEFINE_string('my_cnn', 'mnist',"""模型的名称""")
# np.empty((s,d*f))
[X_train, y_train] = read.read_and_decode('/home/r/mycar/train.tfrecords',10000)
print(X_train.shape)
print(y_train.shape)
[X_val, y_val] = read.read_and_decode('/home/r/mycar/Validation.tfrecords',3475)
def chooseone(image,label,batchsize):
    q = image.shape[0]
    im = np.empty((batchsize,1152))
    la = np.empty((batchsize,36))
    for i in range(batchsize):
        a = random.randint(0,q-1)
        im[i]=image[a]
        la[i]=label[a]
    return im,la

# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义卷积层
def conv2d(x, W):
    # 默认 strides[0]=strides[3]=1, strides[1]为x方向步长，strides[2]为y方向步长
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# pooling 层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# 模型文件所在的文件夹，是否存在，如果不存在，则创建文件夹
ckpt = tf.train.latest_checkpoint(FLAGS.my_list)
# ckpt = tf.train.latest_checkpoint(my_list)
if not ckpt:
    if not os.path.exists(FLAGS.my_list):
        os.mkdir(FLAGS.my_list)
X_ = tf.placeholder(tf.float32, [None, 1152])
y_ = tf.placeholder(tf.float32, [None, 36])

# 把X转为卷积所需要的形式
X = tf.reshape(X_, [-1,48, 24, 1])
# 第一层卷积：3×3×1卷积核32个 [3，3，1，32],h_conv1.shape=[-1, 48, 24, 32],学习32种特征
W_conv1 = weight_variable([3,3,1,48])
b_conv1 = bias_variable([48])
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

# 第一个pooling 层[-1, 48, 24, 32]->[-1, 24, 12, 32]
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积：5×5×32卷积核64个 [3，3，32，64],h_conv2.shape=[-1, 24, 12, 64]
W_conv2 = weight_variable([5,5,48,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第二个pooling 层,[-1, 24, 12, 64]->[-1, 14, 6, 64] 
h_pool2 = max_pool_2x2(h_conv2)

# 
W_conv3 = weight_variable([3,3,64,96])
b_conv3 = bias_variable([96])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# flatten层，[-1, 7, 7, 64]->[-1, 7*7*64],即每个样本得到一个7*7*64维的样本
h_pool2_flat = tf.reshape(h_pool3, [-1, 6*3*96])

# fc1
W_fc1 = weight_variable([6*3*96, 512])
b_fc1 = bias_variable([512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([512, 36])
b_fc2 = bias_variable([36])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 1.损失函数：cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv)) 
# 2.优化函数：AdamOptimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 3.预测准确结果统计
#　预测值中最大值（１）即分类结果，是否等于原始标签中的（１）的位置。argmax()取最大值所在的下标
z = tf.argmax(y_conv, 1)
q = tf.arg_max(y_, 1)
correct_prediction = tf.equal(z, q)  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_acc_sum = tf.Variable(0.0)
batch_acc = tf.placeholder(tf.float32)
new_test_acc_sum = tf.add(test_acc_sum, batch_acc)
update = tf.assign(test_acc_sum, new_test_acc_sum)
saver=tf.train.Saver(max_to_keep=2)
# 定义了变量必须要初始化，或者下面形式
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(FLAGS.my_list,sess.graph)

    ckpt = tf.train.latest_checkpoint(FLAGS.my_list)
    step=0
    if ckpt:
        saver.restore(sess=sess,save_path=ckpt)
        step = int(ckpt[len(os.path.join(FLAGS.my_list, FLAGS.my_cnn)) + 1:])

    #     check_point_path = '/home/r/mycar/cnn' # 保存好模型的文件路径
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
    #     saver.restore(sess,ckpt.model_checkpoint_path)

    Y = np.zeros(3000)
#     X_batch, y_batch = mnist.test.next_batch(batch_size=10000)
    X_batch,y_batch = chooseone(X_val, y_val,3000)
    Ytemp = y_conv.eval(feed_dict={X_: X_batch, keep_prob: 1.0})
    for i in range(3000):
            #生成0-9标签
        Y[i] = np.argmax(Ytemp[i])
        # print(Y[i])
    # print("test accuracy %g" % accuracy.eval(feed_dict={X_: X_batch, y_: y_batch, keep_prob: 1.0}))
fp = open("test.txt", "w+")
for i in range(3000):
    fp.write(str(int(Y[i]))+"\n")
fp.close()
# # 训练
#     for i in range(3000):
#         X_batch,y_batch = chooseone(X_train,y_train,50)
#         # X_batch, y_batch = mnist.train.next_batch(batch_size=50)
#         if i % 100 == 0:
#             train_accuracy = accuracy.eval(feed_dict={X_: X_batch, y_: y_batch, keep_prob: 1.0})
#             print ("step %d, training acc %g" % (i, train_accuracy))
#             ckptname=os.path.join(FLAGS.my_list, FLAGS.my_cnn)
#             saver.save(sess,ckptname,global_step=i)
#         train_step.run(feed_dict={X_: X_batch, y_: y_batch, keep_prob: 0.5})  
    
    
# 全部训练完了再做测试，batch_size=100
    