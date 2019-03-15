import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#플레이스홀더 초기화 & 학습 후 예측을 위한 플레이스홀더 초기화 추가
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # [입력 데이터 수, 28 X 28 사이즈, 차원의 특징 수(회색조 데이터)]
Y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)

#layer를 이용한 신경망 구현(X, 커널 수, 커널 사이즈, 활성화 함수)
L1 = tf.layers.conv2d(X, 32, [3, 3], activation=tf.nn.relu, padding='SAME') # padding : 보다 정확한 테두리 값 평가를 위해 외각을 한칸 밖으로 움직이는 옵션
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2], padding='SAME')
L1 = tf.layers.dropout(L1, 0.7, is_training)

L2 = tf.layers.conv2d(L1, 64, [3, 3], activation=tf.nn.relu, padding='SAME')
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2], padding='SAME')
L2 = tf.layers.dropout(L2, 0.7, is_training)

# fully connected layer 구현
L3 = tf.layers.flatten(L2)
L3 = tf.layers.dense(L3, 128, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.5, is_training)

model = tf.layers.dense(L3, 10, activation=None)

#
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#모델 초기화 및 학습 진행
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# MNIST 15번 학습
for epoch in range(10):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #데이터를 28 * 28 형태로 재구성
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, is_training : True})

        total_cost += cost_val

        # 학습하는 세대의 평균 손실값 출력
    print('Epoch : ', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.4f}'.format(total_cost / total_batch))

print('최적화 완료!')

# 결과 확인
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : ', sess.run(accuracy, feed_dict={X : mnist.test.images.reshape(-1, 28, 28, 1), Y : mnist.test.labels, is_training : False}))