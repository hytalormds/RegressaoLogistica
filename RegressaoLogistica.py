import tensorflow as tf
from tqdm import trange
from tensorflow.examples.tutorials.mnist import input_data

# Importando os Dados
mnist = input_data.read_data_sets("datasets/MNIST_data/", one_hot=True)

# Criando o Modelo
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Definir Perda e Otimizador
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Criar um Objeto Session, Inicializar Todas as Variáveis

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Treinamento
for _ in trange(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Modelo Treinado de Teste
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Test accuracy: {0}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

sess.close()
