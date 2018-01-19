import numpy as np
import tensorflow as tf
import matplotlib as mpl
import os
import pylab
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from tensorflow.python.ops import array_ops as array_ops_
import scipy.io as sio
import matplotlib.pyplot as plt
import math

mpl.use('Agg')
from matplotlib import pyplot as plt
# 初始化
learn = tf.contrib.learn

HIDDEN_SIZE = 20  # LSTM中隐藏节点的个数
NUM_LAYERS = 2  # LSTM的层数

TIMESTEPS = 5  # 循环神经网络的阶段长度
TRAINING_STEPS = 2000  # 训练轮数
BATCH_SIZE = 32  # 32

TRAINING_EXAMPLES = 512  # 训练数据个数
TESTING_EXAMPLES = 256  # 测试数据个数

def generate_data(seq):
    X = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入
    for i in range(len(seq) - 5):
        X.append(seq[i : i + 5])
    return np.array(X, dtype=np.float32)

def lstm_model(X, y):

    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    #dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=0.98)
    #cell = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm] * NUM_LAYERS)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
    x_ = tf.unpack(X, axis=1)

    output, _ = tf.nn.rnn(cell, x_, dtype=tf.float32)
    # 在本问题中值关注最后一个时刻的输出结果，该结果为下一时刻的预测值
    output = output[-1]
    #prediction = []
    #loss = []
    # 平方差损失函数
    prediction, loss = learn.models.linear_regression(output, y)
    #prediction2, loss2 = learn.models.linear_regression(output[1], y[1])
    #prediction.append([prediction1,prediction2])
    #loss.append(loss1+loss2)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer="Adagrad",learning_rate=0.1)

    '''lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])

    # 通过无激活函数的全联接层计算线性回归，并将数据压缩成一维数组的结构。
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])

    loss = tf.losses.mean_squared_error(predictions, labels)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)'''
    return prediction, loss, train_op

def main():
    #os.makedirs('result')
    # 建立深层循环网络模型
    #regressor_abs = learn.Estimator(model_fn=lstm_model)
    regressor_abs = SKCompat(learn.Estimator(model_fn=lstm_model,model_dir="Models/model_2"))
    # load data
    #matfn = 'D:\\MATLAB\\work\\fitting\\after full-duplex data\\full-duplex.mat'  # the path of .mat data
    matfn = 'result/full-duplex.mat'
    data = sio.loadmat(matfn)

    for j in [0,5,10,15,20,25,30,34]:
        x = data['x'+str(j)+'b']
        y = data['y'+str(j)+'b']
        Error = []
        for seqnum in range(100):
            print("", seqnum)
            complex_x = []
            complex_y = []

            for i in range(len(x[seqnum])):
                x_abs = abs(x[seqnum][i])
                x_w = math.atan(x[seqnum][i].imag / x[seqnum][i].real)
                complex_x.append([x_abs, x_w])

            for i in range(len(y[seqnum])):
                y_abs = abs(y[seqnum][i])
                y_w = math.atan(y[seqnum][i].imag / y[seqnum][i].real)
                complex_y.append([y_abs, y_w])

    ####for abs
            train_X_abs = generate_data(complex_x[0:TRAINING_EXAMPLES])
            #train_y_abs = complex_y[int(TIMESTEPS / 2):TRAINING_EXAMPLES + int(TIMESTEPS / 2) - 1]
            train_y_abs = complex_y[TIMESTEPS-1:TRAINING_EXAMPLES]
            test_X_abs = generate_data(complex_x[TRAINING_EXAMPLES:TESTING_EXAMPLES + TRAINING_EXAMPLES - 1])
            #test_y_abs = complex_y[TRAINING_EXAMPLES + int(TIMESTEPS / 2):TESTING_EXAMPLES + TRAINING_EXAMPLES + int(TIMESTEPS / 2) - 1]
            test_y_abs = complex_y[TRAINING_EXAMPLES :TESTING_EXAMPLES + TRAINING_EXAMPLES - 1]
            # 调用fit函数训练模型
            regressor_abs.fit(train_X_abs, train_y_abs, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
            #regressor_abs.fit(train_X_abs, train_y_abs, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
    # 预测
            predicted_abs = [[pred] for pred in regressor_abs.predict(test_X_abs)]
    # db
            a_real = []
            b_real = []
            a_pre = []
            b_pre = []
            for i in range(len(test_y_abs)):
                a_real.append(test_y_abs[i][0] * math.cos(math.atan(test_y_abs[i][1])))
                b_real.append(test_y_abs[i][0] * math.sin(math.atan(test_y_abs[i][1])))
                a_pre.append(np.array(predicted_abs[i])[0][0] * math.cos(math.atan(np.array(predicted_abs[i])[0][1])))
                b_pre.append(np.array(predicted_abs[i])[0][0] * math.sin(math.atan(np.array(predicted_abs[i])[0][1])))
            diff_real = []
            diff_imag = []
            for i in range(len(a_real)):
                diff_real.append(a_real[i] - a_pre[i])
                diff_imag.append(b_real[i] - b_pre[i])
            error = 10 * math.log10(np.mean(np.square(predicted_abs))) - 10 * math.log10(np.mean(np.square(diff_real) + np.square(diff_imag)))
            print("db is : %f" % error)
            Error.append(error)
        sio.savemat('result/' + str(j) + 'DB.mat', {'Error': Error})

# plot
'''pylab.figure(1)  # 创建图表1

for i in range(100):
    pylab.plot(i, np.array(predicted_abs[i])[0][0], 'r*')
    pylab.plot(i, test_y_abs[i][0], 'bo')
pylab.show()

pylab.figure(2)  # 创建图表2
for i in range(100):
    pylab.plot(i, np.array(predicted_abs[i])[0][1], 'r*')
    pylab.plot(i, test_y_abs[i][1], 'bo')
pylab.show()'''
if __name__ == '__main__':
    main()