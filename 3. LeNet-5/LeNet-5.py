import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets  # 导入经典数据集加载模块
from tensorflow.keras import Sequential
from tensorflow.keras import layers,losses, optimizers

# 加载 MNIST 数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()   # 返回数组的形状
# 将数据集转换为DataSet对象，不然无法继续处理
train_db = tf.data.Dataset.from_tensor_slices((x, y))
# 将数据顺序打散
train_db = train_db.shuffle(10000)  # 数字为缓冲池的大小
# 设置批训练
train_db = train_db.batch(512)  # batch size 为 128


#预处理函数
def preprocess(x, y):   # 输入x的shape 为[b, 32, 32], y为[b]
    # 将像素值标准化到 0~1区间
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 将图片改为28*28大小的
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)  # 转成整型张量
    y = tf.one_hot(y, depth=10)
    return x, y
# 将数据集传入预处理函数，train_db支持map映射函数


train_db = train_db.map(preprocess)
# 训练20个epoch
train_db = train_db.repeat(20)
# 以同样的方式处理测试集
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(512).map(preprocess)
network = Sequential([
    layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层, 6 个 3x3 卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积层, 16 个 3x3 卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Flatten(),  # 打平层，形成一维向量，方便全连接层处理
    layers.Dense(120, activation='relu'),  # 全连接层， 120 个节点
    layers.Dense(84, activation='relu'),  # 全连接层， 84 节点
    layers.Dense(10)  # 全连接层， 10 个节点
    ])
    # 构建网络模型，给输入X的形状，其中4为随意的BatchSize
network.build(input_shape=(4, 28, 28, 1))
    # 统计网络信息
    # print(network.summary())
optimizer = optimizers.Adam(lr=1e-4)
loss_all = []
    # 创建损失函数的类，在实际计算时直接调用类实例即可
criteon = losses.CategoricalCrossentropy(from_logits=True)
for step, (x,  y) in enumerate(train_db):
        # 将输入张量x的shape[512.784]变成[
    x = tf.reshape(x, (-1, 28, 28))
    with tf.GradientTape() as tape:
            # 插入通道维度，=>[b,28,28,1]
        x = tf.expand_dims(x, axis=3)
            # 前向计算，获得10类别的预测分布，[b, 784] => [b, 10]
        out = network(x)
            # 将真实标签转化为one-hot编码，[b] => [b, 10]
            # 计算交叉熵损失函数，标量
        loss = criteon(y, out)
        # 自动计算梯度，关键看如何表示待优化变量
    grads = tape.gradient(loss, network.trainable_variables)
        # 自动更新参数
    optimizer.apply_gradients(zip(grads, network.trainable_variables))
        # step为80次时，记录并输出损失函数结果
    if step % 100 == 0:
        print(step, 'loss:', float(loss))
        loss_all.append(float(loss))
        # step为80次时，用测试集验证模型
    if step % 100 == 0:
        total, total_correct = 0., 0
        correct, total = 0, 0
        for x, y in test_db:  # 遍历所有训练集样本
                # 插入通道维度，=>[b,28,28,1]
            x = tf.reshape(x, (-1, 28, 28))
            x = tf.expand_dims(x, axis=3)
                # 前向计算，获得10类别的预测分布，[b, 784] => [b, 10]
            out = network(x)
                # 真实的流程时先经过softmax，再argmax
                # 但是由于softmax不改变元素的大小相对关系，故省去
            pred = tf.argmax(out, axis=-1)
            y = tf.cast(y, tf.int64)
            y = tf.argmax(y, axis=-1)
                # 统计预测正确数量
            correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
                # 统计预测样本总数
            total += x.shape[0]
            # 计算准确率
        print('test acc:', correct / total)

