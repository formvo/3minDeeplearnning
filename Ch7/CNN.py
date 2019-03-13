import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#플레이스홀더 초기화 & 학습 후 예측을 위한 플레이스홀더 초기화 추가
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # [입력 데이터 수, 28 X 28 사이즈, 차원의 특징 수(회색조 데이터)]
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#신경망 구현 & 드랍아웃 코드 추가(레이어, 뉴런의 비율)
W1 = tf.Variable(tf.random_normal([3, 3, 1 ,32], stddev=0.01)) # [3 X 3 커널 사이즈 , 1개의 특징, 32개의 커널]
L1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding='SAME') # padding : 보다 정확한 테두리 값 평가를 위해 외각을 한칸 밖으로 움직이는 옵션
L1 = tf.nn.relu(L1)
# 풀링 레이어
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32 ,64], stddev=0.01)) # [3 X 3 커널 사이즈, 앞에서 구성한 컨볼루션 계층, 컨볼루션 계층에서 찾아낸 특징 수]
L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding='SAME')

# fully connected layer 구현
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

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
for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #데이터를 28 * 28 형태로 재구성
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})

        total_cost += cost_val

        # 학습하는 세대의 평균 손실값 출력
        print('Epoch : ', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

# 결과 확인
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : ', sess.run(accuracy, feed_dict={X : mnist.test.images.reshape(-1, 28, 28, 1), Y : mnist.test.labels, keep_prob : 1}))