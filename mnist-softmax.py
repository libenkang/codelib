import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/',one_hot=True)

x = tf.placeholder(tf.float32,[None,784])

W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32,[None,10])


#交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
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

"""
使用此模型方案：
    读取一张图片
    把图片转换成（784，）向量常量x_
    out=tf.matmul(x_,W)+b(1,784) (784,10) (1,10)
"""
num=0
for i in range(100):
    image_array = mnist.test.images[i,:]
    image_array =image_array.reshape(1,784)
    out=tf.matmul(image_array,W)+b
    if(tf.argmax(out,1)[0].eval() != tf.argmax(mnist.test.labels[i,:]).eval()):
        print(i)
        num=num+1
print('识别错误的图片：'，num)
#print(tf.argmax(out,1)[0].eval())
#print(tf.argmax(mnist.test.labels[i,:]).eval())
#print(mnist.test.labels[i,:])



