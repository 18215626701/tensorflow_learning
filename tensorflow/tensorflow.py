test
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
train_x=np.linspace(-1, 1, 100)
train_y=2*train_x + np.random.randn(*train_x.shape)*0.3
plt.plot(train_x, train_y, "ro", label='Original data')
plt.legend()
#plt.show()
X=tf.placeholder("float")
Y=tf.placeholder("float")
W=tf.Variable(tf.random_normal([1]), name="weight")
b=tf.Variable(tf.zeros([1], name="bias"))
z=tf.multiply(X, W) + b
const=tf.reduce_mean(tf.square(Y-z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(const)
init = tf.global_variables_initializer()
traininig_epochs = 20
display_step = 2

with tf.Session() as sess:
    sess.run(init)
    plotdata = {"batchsize":[], "loss":[]}
    for epoch in range(traininig_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X:x, Y:y})

        if epoch % display_step == 0:
            loss = sess.run(const, feed_dict = {X:train_x, Y:y})
            print("Epoch:", epoch + 1, "const=", loss, "W=", sess.run(W),  "b=", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"] .append(loss)

    print("Finished!")
    print("const=", sess.run(const, feed_dict={X:train_x, Y:train_y}), "W=", sess.run(W), "b=", sess.run(b))

    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fittedline')
    plt.legend()
    plt.show()
