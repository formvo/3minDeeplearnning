import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#mnist 데이터셋 읽기
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#플레이스홀더 초기화 & 학습 후 예측을 위한 플레이스홀더 초기화 추가
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#신경망 구현 & 드랍아웃 코드 추가(레이어, 뉴런의 비율)
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

#cost 계산 & 손실값 최적화
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#모델 초기화 및 학습 진행
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# MNIST 15번 학습
for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X : batch_xs, Y : batch_ys, keep_prob : 0.8})

        total_cost += cost_val

        # 학습하는 세대의 평균 손실값 출력
        print('Epoch : ', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

# 결과 확인
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : ', sess.run(accuracy, feed_dict={X : mnist.test.images, Y : mnist.test.labels, keep_prob : 1}))