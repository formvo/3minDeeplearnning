import tensorflow as tf

#실제 x, y 데이터
x_data = [1, 2, 3]
y_data = [1, 2, 3]

#가중치와 편향 값을 위한 변수 설정, -1.0 ~ 1.0 사이의 균등분포를 가진 랜덤값으로 초기화
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#학습에 사용될 X,Y 플레이스홀더 설정
X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")

#선형관계 분석을 위한 수식
hypothesis = W * X + b

#실제값과 예측한 값의 차이
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#경사하강법을 이용한 최적화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        _, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n ====== Test ======")
    print("X: 5, Y: ", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5 Y:", sess.run(hypothesis, feed_dict={X: 2.5}))