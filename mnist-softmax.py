import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/',one_hot=True)

x = tf.placeholder(tf.float32,[None,784])

W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32,[None,10])


#交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
#初始化所有变量（在session中保存变量初始值）
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

#正确的预测结果
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#计算预测准确率，都是tensor
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))#0.9185
print(type(mnist))

