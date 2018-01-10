import numpy as np
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import os
import read
import random
# mnist = input_data.read_data_sets('data/fashion', one_hot=True)
def cnn():
    # FLAGS = tf.app.flags.FLAGS
    # tf.app.flags.DEFINE_string('my_list', './cnn', """存放模型的目录""")

    # tf.app.flags.DEFINE_string('my_cnn', 'capcha', """模型的名称""")
    [X_test, y_test] = read.read_and_decode('./data/test.tfrecords', 1000)
    [X_val, y_val] = read.read_and_decode('./data/validation.tfrecords', 2000)



    def chooseone(image, label, batchsize):
        q = image.shape[0]
        im = np.empty((batchsize, 2240))
        la = np.empty((batchsize, 44))
        for i in range(batchsize):
            a = random.randint(0, q-1)
            im[i] = image[a]
            la[i] = label[a]
        return im, la
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
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # pooling 层


    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    # 模型文件所在的文件夹，是否存在，如果不存在，则创建文件夹
    ckpt = tf.train.latest_checkpoint('./cnn')
    # # ckpt = tf.train.latest_checkpoint(my_list)
    if not ckpt:
        if not os.path.exists('./cnn'):
            os.mkdir('./cnn')
    X_ = tf.placeholder(tf.float32, [None, 2240])
    y_ = tf.placeholder(tf.float32, [None, 44])
    y_1 = tf.placeholder(tf.float32, [None, 11])
    y_2 = tf.placeholder(tf.float32, [None, 11])
    y_3 = tf.placeholder(tf.float32, [None, 11])
    y_4 = tf.placeholder(tf.float32, [None, 11])
    keep_prob = tf.placeholder(tf.float32)
    # 把X转为卷积所需要的形式
    with tf.name_scope("reshape1"):
        X = tf.reshape(X_, [-1, 40, 56, 1])
    # 第一层卷积：3×3×1卷积核32个 [3，3，1，32],h_conv1.shape=[-1, 40, 56, 32],学习32种特征
    with tf.name_scope("conv1_1"):
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32]) 
        h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

    with tf.name_scope("pool1"):
        # 第一个pooling 层[-1, 40, 56, 32]->[-1, 20, 28, 32]
        h_pool1 = max_pool_2x2(h_conv1)
    
    with tf.name_scope("conv1_2"):
         # 第二层卷积：5×5×32卷积核64个 [3，3，32，48],h_conv2.shape=[-1, 20, 28, 48]
        W_conv1_ = weight_variable([3, 3, 32, 48])
        b_conv1_ = bias_variable([48]) 
        h_conv1_ = tf.nn.relu(conv2d(h_pool1, W_conv1_) + b_conv1_)

    # 第三层卷积：5×5×32卷积核64个 [3，3，32，64],h_conv2.shape=[-1, 20, 28, 64]
    with tf.name_scope("conv2_1"):
        W_conv2 = weight_variable([3, 3, 48, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1_, W_conv2) + b_conv2)

    with tf.name_scope("pool2"):
    # 第二个pooling 层,[-1, 20, 28, 64]->[-1, 10, 14, 64] 
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope("conv3_1"):
    # 第三层卷积：5×5×32卷积核64个 [3，3，64，96],h_conv2.shape=[-1, 20, 28, 96]
        W_conv3 = weight_variable([3, 3, 64, 96])
        b_conv3 = bias_variable([96])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    with tf.name_scope("pool3"):
        # 第三个pooling 层,[-1, 10, 14, 96]->[-1, 5, 7, 96] 
        h_pool3 = max_pool_2x2(h_conv3)

    # flatten层，[-1, 5, 7, 96]->[-1, 5*7*96],即每个样本得到一个7*7*64维的样本
    with tf.name_scope("flatting"):
        h_pool2_flat = tf.reshape(h_pool3, [-1, 5*7*96])
    with tf.name_scope("fc1"):
    # fc1
        W_fc1 = weight_variable([5*7*96, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
    # keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope("dropout"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    with tf.name_scope("y_cov1"):
        W_fc2 = weight_variable([1024, 11])
        b_fc2 = bias_variable([11])
        y_conv1 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    with tf.name_scope("y_cov2"):
        W_fc3 = weight_variable([1024, 11])
        b_fc3 = bias_variable([11])
        y_conv2 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)
    with tf.name_scope("y_cov3"):
        W_fc4 = weight_variable([1024, 11])
        b_fc4 = bias_variable([11])
        y_conv3 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc4) + b_fc4)
    with tf.name_scope("y_cov4"):
        W_fc5 = weight_variable([1024, 11])
        b_fc5 = bias_variable([11])
        y_conv4 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc5) + b_fc5)
    # 1.损失函数：cross_entropy
    with tf.name_scope("cross_entropy"):
        cross_entropy1 = -tf.reduce_sum(y_1 * tf.log(y_conv1), name='cross_entropy_1') 
        cross_entropy2 = -tf.reduce_sum(y_2 * tf.log(y_conv2), name='cross_entropy_2') 
        cross_entropy3 = -tf.reduce_sum(y_3 * tf.log(y_conv3), name='cross_entropy_3') 
        cross_entropy4 = -tf.reduce_sum(y_4 * tf.log(y_conv4), name='cross_entropy_4') 
        cross_entropy = (cross_entropy1+cross_entropy2+cross_entropy3+cross_entropy4)/4
        tf.summary.scalar('loss_1', cross_entropy1)  
        tf.summary.scalar('loss_2', cross_entropy2)
        tf.summary.scalar('loss_3', cross_entropy3)
        tf.summary.scalar('loss_4', cross_entropy4)
        # tf.summary.scalar('loss', cross_entropy)
        merged = tf.summary.merge_all() 
    # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y_))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y_))
    # 2.优化函数：AdamOptimizer
    with tf.name_scope("Adam"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # 3.预测准确结果统计
    # 预测值中最大值（１）即[分类结果，是否等于原始标签中的（１）的位置。argmax()取最大值所在的下标
    with tf.name_scope("accuracy"):
        y_conva = tf.concat([y_conv1, y_conv2], 1) 
        y_convb = tf.concat([y_conv3, y_conv4], 1)
        y_conv = tf.concat([y_conva, y_convb], 1)
        predict = tf.reshape(y_conv, [-1, 4, 11])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(y_, [-1, 4, 11]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(tf.cast(tf.reduce_mean(tf.cast(correct_pred,tf.float32),1),tf.int64),tf.float32))
        # accuracy = tf.floor(accuracy
        correct_prediction1 = tf.equal(tf.argmax(y_conv1, 1), tf.arg_max(y_1, 1))  # 
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
        correct_prediction2 = tf.equal(tf.argmax(y_conv2, 1), tf.arg_max(y_2, 1))  
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
        correct_prediction3 = tf.equal(tf.argmax(y_conv3, 1), tf.arg_max(y_3, 1))  
        accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
        correct_prediction4 = tf.equal(tf.argmax(y_conv4, 1), tf.arg_max(y_4, 1))  
        accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))
    test_acc_sum = tf.Variable(0.0)
    batch_acc = tf.placeholder(tf.float32)
    new_test_acc_sum = tf.add(test_acc_sum, batch_acc)
    update = tf.assign(test_acc_sum, new_test_acc_sum)
    saver = tf.train.Saver(max_to_keep=2)
    train_writer = tf.summary.FileWriter('graphs/')
    train_writer.add_graph(tf.get_default_graph())
    # 定义了变量必须要初始化，或者下面形式
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # writer = tf.summary.FileWriter(FLAGS.my_list, sess.graph)
        ckpt = tf.train.latest_checkpoint('./cnn')
        step = 0
        if ckpt:
            saver.restore(sess=sess, save_path=ckpt)
            step = int(ckpt[len(os.path.join('./cnn', 'capcha')) + 1:])
        
   
    # 全部训练完了再做测试，batch_size=100
        for i in range(100): 
            X_batch, y_batch = chooseone(X_test, y_test, 100)
            # X_batch, y_batch = mnist.test.next_batch(batch_size=100)
            test_acc = accuracy.eval(feed_dict={X_: X_batch, y_: y_batch, y_1: y_batch[:, :11], y_2: y_batch[:, 11:22], y_3: y_batch[:, 22:33], y_4: y_batch[:, 33:44], keep_prob: 1.0})
            summary, train_accuracy1, train_accuracy2, train_accuracy3, train_accuracy4 = sess.run([merged,accuracy1, accuracy2, accuracy3, accuracy4], feed_dict={X_: X_batch, y_: y_batch, y_1: y_batch[:, :11], y_2: y_batch[:, 11:22], y_3: y_batch[:, 22:33], y_4: y_batch[:, 33:44], keep_prob: 1.0})
            update.eval(feed_dict={batch_acc: test_acc})
            if (i+1) % 100 == 0:
                print('training1 acc %g,training1 acc %g,training1 acc %g,training1 acc %g' % (train_accuracy1, train_accuracy2, train_accuracy3, train_accuracy4))
                print("testing step %d, test_acc_sum %g" % (i+1, test_acc_sum.eval()))
        d=test_acc_sum.eval() / 100.0
    return d

cnn()