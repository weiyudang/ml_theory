# Multitask-Learning(MTL)

### 1.详解谷歌之多任务学习模型MMoE(KDD 2018)

**一、Motivation**

多任务模型通过学习不同任务的联系和差异，可提高每个任务的学习效率和质量。多任务学习的的框架广泛采用**shared-bottom**的结构，不同任务间共用底部的隐层。这种结构本质上**可以减少过拟合的风险**，但是效果上可能受到任务差异和数据分布带来的影响。也有一些其他结构，比如两个任务的参数不共用，但是通过对不同任务的参数增加**L2范数**的限制；也有一些对每个任务分别学习一套隐层然后学习所有隐层的组合。和shared-bottom结构相比，这些模型对增加了针对任务的特定参数，在任务差异会影响公共参数的情况下对最终效果有提升。缺点就是模型增加了参数量所以需要更大的数据量来训练模型，而且模型更复杂并不利于在真实生产环境中实际部署使用。

**二、模型介绍**

**MMoE**模型的结构(下图c)**基于广泛使用的Shared-Bottom结构(下图a)和MoE结构**，其中图(b)是图(c)的一种特殊情况，下面依次介绍。

![img](https://pic4.zhimg.com/80/v2-c578c5e506e202483c64fdac94281d4f_1440w.jpg)



- **Shared-Bottom Multi-task Model**

如上图a所示，**shared-bottom**网络（表示为函数$f$）位于底部，多个任务共用这一层。往上，K个子任务分别对应一个**tower network**（表示为![[公式]](https://www.zhihu.com/equation?tex=h%5Ek) ），每个子任务的输出 ![[公式]](https://www.zhihu.com/equation?tex=y_k%3Dh%5Ek%28f%28x%29%29) 

- **Mixture-of-Experts**

  **MoE**模型可以形式化表示为 ![[公式]](https://www.zhihu.com/equation?tex=y%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bg%28x%29_if_i%28x%29%7D) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bg%28x%29_i%7D%3D1) 。其中 ![[公式]](https://www.zhihu.com/equation?tex=f_i%2Ci%3D1%2C%5Ccdots%2Cn) 是n个expert network（**expert network**可认为是一个神经网络）。g是组合experts结果的gating network，具体来说g产生n个experts上的概率分布，最终的输出是**所有experts的带权加和**。显然，MoE可看做基于多个独立模型的集成方法。这里注意MoE并不对应上图中的b部分。

  ```
  理解：相对shared-bottom  Moe使用多个export net 对特征进行学习，然后再通过gating net 控制不同的export 在最终目标决策中占的比重；由于每个任务都有自己的gating net ,所有通过控制gating net 实现对export net 的选择性利用。不同任务的gating network 可以学习到不同的组合的exports的模式，因此模型考虑到了任务的相关性和区别。
  ```

  后面有些文章将MoE作为一个基本的组成单元，将多个MoE结构堆叠在一个大网络中。比如一个MoE层可以接受上一层MoE层的输出作为输入，其输出作为下一层的输入使用。

- **所提模型Multi-gate Mixture-of-Experts**

  文章提出的模型（简称MMoE）目的就是**相对于shared-bottom结构不明显增加模型参数的要求下捕捉任务的不同**。其核心思想是将**shared-bottom网络中的函数f替换成MoE层**，如上图c所示，形式化表达为：

  ![[公式]](https://www.zhihu.com/equation?tex=y_k%3Dh%5Ek%28f%5Ek%28x%29%29%2Cf%5Ek%28x%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bg%5Ek%28x%29_if_i%28x%29%7D)

  其中 ![[公式]](https://www.zhihu.com/equation?tex=g%5Ek%28x%29%3Dsoftmax%28W_%7Bgk%7Dx%29) ，输入就是input feature，输出是所有experts上的权重。

  一方面，因为gating networks通常是轻量级的，而且expert networks是所有任务共用，所以相对于论文中提到的一些baseline方法在计算量和参数量上具有优势。

  另一方面，相对于所有任务公共一个门控网络(One-gate MoE model，如上图b)，这里MMoE(上图c)中每个任务使用单独的gating networks。**每个任务的gating networks通过最终输出权重不同实现对experts的选择性利用**。**不同任务的gating networks可以学习到不同的组合experts的模式**，因此模型考虑到了捕捉到任务的相关性和区别。

**三、总结**

整体来看，这篇文章是对多任务学习的一个扩展，**通过门控网络的机制来平衡多任务**的做法在真实业务场景中具有借鉴意义。下面补充介绍文中的一个数据集设置的做法和实验结果中对不同模型的相互对比分析。



- **人工构造数据集**

在真实数据集中我们无法改变任务之间的相关性，所以不太方便进行研究任务相关性对多任务模型的影响。轮文中人工构建了两个回归任务的数据集，然后通过两个任务的标签的Pearson相关系数来作为任务相关性的度量。在工业界中**通过人工构造的数据集来验证自己的假设**是个有意思的做法。

- **模型的可训练性**

模型的**可训练性，就是模型对于超参数和初始化是否足够鲁棒**。作者在人工合成数据集上进行了实验，观察不同随机种子和模型初始化方法对loss的影响。这里简单介绍下两个现象：第一，Shared-Bottom models的效果方差要明显大于基于MoE的方法，说明Shared-Bottom模型有很多偏差的局部最小点；第二，如果任务相关度非常高，则OMoE和MMoE的效果近似，但是如果任务相关度很低，则OMoE的效果相对于MMoE明显下降，说明**MMoE中的multi-gate的结构对于任务差异带来的冲突有一定的缓解**作用。

**rerference**

- [mmoe code](https://github.com/drawbridge/keras-mmoe/blob/master/mmoe.py)
- [Smile detection, gender and age estimation using Multi-task Learning](https://github.com/truongnmt/multi-task-learning)

### 2. ESMM

本文介绍 阿里妈妈团队 发表在 SIGIR’2018 的论文《[Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1804.07931)》。文章基于 Multi-Task Learning 的思路，提出一种新的CVR预估模型——ESMM，有效解决了真实场景中CVR预估面临的数据稀疏以及样本选择偏差这两个关键问题。 实践出真知，论文一点也不花里胡哨，只有4页，据传被 SIGIR’2018 高分录用。

**一、Motivation**

不同于CTR预估问题，CVR预估面临两个关键问题：

1. **Sample Selection Bias (SSB)** 转化是在点击之后才“有可能”发生的动作，传统CVR模型通常以点击数据为训练集，其中点击未转化为负例，点击并转化为正例。但是训练好的模型实际使用时，则是对整个空间的样本进行预估，而非只对点击样本进行预估。即是说，训练数据与实际要预测的数据来自不同分布，这个偏差对模型的泛化能力构成了很大挑战。
2. **Data Sparsity (DS)** 作为CVR训练数据的点击样本远小于CTR预估训练使用的曝光样本。

一些策略可以缓解这两个问题，例如从曝光集中对unclicked样本抽样做负例缓解SSB，对转化样本过采样缓解DS等。但无论哪种方法，都没有很elegant地从实质上解决上面任一个问题。

可以看到：点击—>转化，本身是两个强相关的连续行为，作者希望在模型结构中显示考虑这种“行为链关系”，从而可以在整个空间上进行训练及预测。这涉及到CTR与CVR两个任务，因此使用多任务学习（MTL）是一个自然的选择，论文的关键亮点正在于“如何搭建”这个MTL。

**二、model**

介绍ESMM之前，我们还是先来思考一个问题——“**CVR预估到底要预估什么**”，论文虽未明确提及，但理解这个问题才能真正理解CVR预估困境的本质。想象一个场景，一个item，由于某些原因，例如**在feeds中的展示头图很丑，它被某个user点击的概率很低，但这个item内容本身完美符合这个user的偏好，若user点击进去，那么此item被user转化的概率极高**。CVR预估模型，预估的正是这个转化概率，**它与CTR没有绝对的关系，很多人有一个先入为主的认知，即若user对某item的点击概率很低，则user对这个item的转化概率也肯定低，这是不成立的。**更准确的说，**CVR预估模型的本质，不是预测“item被点击，然后被转化”的概率**（CTCVR）**，而是“假设item被点击，那么它被转化”的概率**（CVR）。这就是不能直接使用全部样本训练CVR模型的原因，因为咱们压根不知道这个信息：那些unclicked的item，假设他们被user点击了，它们是否会被转化。如果直接使用0作为它们的label，会很大程度上误导CVR模型的学习。

认识到点击（CTR）、转化（CVR）、点击然后转化（CTCVR）是三个不同的任务后，我们再来看三者的关联：

![[公式]](https://www.zhihu.com/equation?tex=%5Cunderbrace%7B+p%28z%5C%26y%3D1+%7C+%5Cbm%7Bx%7D%29+%7D_%7BpCTCVR%7D+%3D+%5Cunderbrace%7B+p%28z%3D1+%7Cy%3D1%2C+%5Cbm%7Bx%7D%29++%7D_%7BpCVR%7D+~+%5Cunderbrace%7B+p%28y%3D1+%7C+%5Cbm%7Bx%7D%29++%7D_%7BpCTR%7D%2C++~~~~~~~~~~~~~~~~~~~~~~~%281%29)

其中 ![[公式]](https://www.zhihu.com/equation?tex=z%2Cy) 分别表示conversion和click。注意到，在全部样本空间中，CTR对应的label为click，而CTCVR对应的label为click & conversion，**这两个任务是可以使用全部样本的**。**那为啥不绕个弯，通过这学习两个任务，再根据上式隐式地学习CVR任务呢？**ESMM正是这么做的，具体结构如下：

![img](https://pic2.zhimg.com/80/v2-d999a47e9ebfcc3fe1b61559b421e2c9_1440w.jpg)

仔细观察上图，留意以下几点：1）**共享Embedding** CVR-task和CTR-task使用相同的特征和特征embedding，即两者从Concatenate之后才学习各自部分独享的参数；2）**隐式学习pCVR** 啥意思呢？这里pCVR（粉色节点）仅是网络中的一个**variable，没有显示的监督信号。**

具体地，反映在目标函数中：

![[公式]](https://www.zhihu.com/equation?tex=L%28%5Ctheta_%7Bcvr%7D%2C+%5Ctheta_%7Bctr%7D%29+%3D+%5Csum_%7Bi%3D1%7D%5E%7BN%7D+l+%28+y_i%2C+f%28%5Cbm%7Bx%7D_i%3B%5Ctheta_%7Bctr%7D%29+%29+%2B+%5Csum_%7Bi%3D1%7D%5E%7BN%7D+l+%28+y_i%5C%26z_i%2C+f%28%5Cbm%7Bx%7D_i%3B%5Ctheta_%7Bctr%7D%29%2Af%28%5Cbm%7Bx%7D_i%3B%5Ctheta_%7Bcvr%7D+%29%29+%EF%BC%8C)

即利用CTCVR和CTR的监督信息来训练网络，隐式地学习CVR，这正是ESMM的精华所在，至于这么做的必要性以及合理性，本节开头已经充分论述了。

再思考下，ESMM的结构是基于“乘”的关系设计——pCTCVR=pCVR*pCTR，是不是也可以通过“除”的关系得到pCVR，即 pCVR = pCTCVR / pCTR ？例如分别训练一个CTCVR和CTR模型，然后相除得到pCVR，其实也是可以的，但这有个明显的缺点：真实场景预测出来的pCTR、pCTCVR值都比较小，“除”的方式容易造成数值上的不稳定。作者在实验中对比了这种方法。

```python
def build_mode(features, mode, params):
  net = fc.input_layer(features, params['feature_columns'])
  # Build the hidden layers, sized according to the 'hidden_units' param.
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
      net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
  # Compute logits
  logits = tf.layers.dense(net, 1, activation=None)
  return logits

def my_model(features, labels, mode, params):
  with tf.variable_scope('ctr_model'):
    ctr_logits = build_mode(features, mode, params)
  with tf.variable_scope('cvr_model'):
    cvr_logits = build_mode(features, mode, params)

  ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
  cvr_predictions = tf.sigmoid(cvr_logits, name="CVR")
  prop = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'probabilities': prop,
      'ctr_probabilities': ctr_predictions,
      'cvr_probabilities': cvr_predictions
    }
    export_outputs = {
      'prediction': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

  y = labels['cvr']
  cvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y, prop), name="cvr_loss")
  ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['ctr'], logits=ctr_logits), name="ctr_loss")
  loss = tf.add(ctr_loss, cvr_loss, name="ctcvr_loss")

  ctr_accuracy = tf.metrics.accuracy(labels=labels['ctr'], predictions=tf.to_float(tf.greater_equal(ctr_predictions, 0.5)))
  cvr_accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prop, 0.5)))
  ctr_auc = tf.metrics.auc(labels['ctr'], ctr_predictions)
  cvr_auc = tf.metrics.auc(y, prop)
  metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
  tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
  tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
  tf.summary.scalar('ctr_auc', ctr_auc[1])
  tf.summary.scalar('cvr_auc', cvr_auc[1])
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN
  optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
```







**rerference**

- [esmm](https://github.com/yangxudong/deeplearning/tree/master/esmm):smile:
- [代码实现](https://zhuanlan.zhihu.com/p/42214716) :smile:
- [阿里CVR预估模型之ESMM](https://zhuanlan.zhihu.com/p/57481330)
- [官媒讲esmm]([https://github.com/alibaba/x-deeplearning/wiki/%E5%85%A8%E7%A9%BA%E9%97%B4%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%A8%A1%E5%9E%8B(ESMM)](https://github.com/alibaba/x-deeplearning/wiki/全空间多任务模型(ESMM)))
- 



### 3.Youtube 排序系统：Recommending What Video to Watch Next

解决`Multitask Learning`, `Selection Bias`这两个排序系统的关键点。

一般推荐系统排序模块的进化路径是：

**ctr任务-> ctr+时长 -> mulittask & selection bias**



多目标，分为两类：

- **engagement objectives**：点击以及于视频观看等参与度目标
- **satisfaction objectives**：Youtube上的喜欢某个视频、打分

真实场景中有很多挑战：

- 有很多不同甚至是冲突的优化目标，比如我们不仅希望用户观看，还希望用户能给出高评价并分享
- 系统中经常有一些隐性偏见，。比如用户是因为视频排得靠前而点击&观看，而非用户确实很喜欢。因此用之前模型产生的数据会引发bias，从而形成一个反馈循环，越来越偏。如何高效有力地去解决还是个尚未解决的问题。

目标基本上就是刚才说的两类，一类 点击、时长， 一类 点赞、打分。分类问题就用cross entropy loss学习，回归问题可就是square loss。最后用融合公式来平衡用户交互和满意度指标，取得最佳效果。

![Figure 2: Replacing shared-bottom layers with MMoE.](http://wd1900.github.io/images/AMRS_f2.png)

**reference**

- [大牛1](http://wd1900.github.io/2019/09/15/Recommending-What-Video-to-Watch-Next-A-Multitask-Ranking-System/)
- 

### 

### 4.Two MTL methods for Deep Learning

#### Hard parameter sharing

<img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\mtl_images-001-2.png" alt="img" style="zoom:50%;" />

- 可以减少过拟合的风险，训练的任务越多，shared layer representation就必须cover 住所有任务

#### Soft parameter sharing



### 5. muti-task learning using uncertainty to weigh losses for scene Geometry and Semantics

本小节主要推导一个多任务损失函数，这个损失函数利用同方差不确定性来最大化高斯似然估计。首先定义一个概率模型：
$$
p(y|f^{W}(x))=N(f^{W}(x),\sigma^2) \tag{2}
$$
这是对回归问题的规律模型定义，$f^{W}(x)$ 是神经网络的输出，$x$是输入数据，$W$是权重，

对于分类问题，同上会将使用softmax 如：
$$
p(y|f^{W}(x))=softmax(f^{W}(x) \tag 3
$$
多任务的似然函数
$$
p(y_{1},····,y_{K}|f^{W}(x))=p(y_{1}|f^{W}(x))...p(y_{K}|f^{W}(x)) \tag 4
$$
其中，$y_{i}$ 是多任务中每个子任务的输出

那么，极大似然估计就可以表示下式，（5）式也表明，该极大似然估计与右边成正比，其中，$\sigma$是高斯分布的标准差，也是作为模型的噪声，接下来的任务就是根据$W$和$\sigma$最大化似然分布

<img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-98b6f6ffb833528b5684187fc6ac092f_720w.png" alt="img" style="zoom:80%;" />

以两个输出y1和y2为例：得到如（6）式高斯分布：

<img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-a18066b842582cb6d914b8ae92402ba5_720w.jpg" alt="img" style="zoom:80%;" />

则此时的极大似然估计为（7）式：

<img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-66c43223c3b317f841efdae456f3c425_720w.jpg" alt="img" style="zoom:80%;" />

可以看到，最后一步中用损失函数替换了y和f的距离计算，即：

<img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-4068c22ad539576138c62a63398ace3d_720w.png" alt="img" style="zoom:80%;" />

同理可知$L_{2}$

**继续分析（7）式子可得，我们的任务是最小化这个极大似然估计，所以，当σ（噪声）增大时，相对应的权重就会降低；另一方面，随着噪声σ减小，相对应的权重就要增加**

接下来，将分类问题也考虑上，分类问题一般加一层softmax，如（8）式所示：

<img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-2fd2931161cd418976b678466565d86c_720w.png" alt="img" style="zoom:80%;" />

那么softmax似然估计为

<img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-a00e622cca3175f72e965cff39c715de_720w.jpg" alt="img" style="zoom:80%;" />

接下来考虑这种情况：模型的两个输出，一个是连续型y1，另一个是独立型y2，分别用高斯分布和softmax分布建模，可得（10）式：

<img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-8900c5b0d73d6a413deb1ff5987b55c4_720w.jpg" alt="img" style="zoom:80%;" />

同理，

<img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-c4816c720e65d6613b48f8c9a4ed463b_720w.png" alt="img" style="zoom:80%;" />

L2（W）替换为：

<img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-28556f3ceac83f0ace7ce7edd2c42878_720w.png" alt="img" style="zoom:80%;" />





#### 关键的概念：

- ### **[不确定性的类型](https://www.leiphone.com/news/201808/P6pPRMWpNt4dHCk0.html)**

  模型不确定性，也就是**认知不确定性**（Epistemic uncertainty）：假设你只有一个数据点，并且你还想知道哪种线性模型最能解释你的数据。但实际情况是，这时你是无法确定哪条线是正确的——我们需要更多的数据！

  <img src="D:\gitrep\ml_theory\ml\mutitask_learning.assets\5b668352892f5.jfif" alt="模型可解释性差？你考虑了各种不确定性了吗？" style="zoom:50%;" />

  左边：数据不足导致了高度不确定性。右边：数据越多不确定性越小。

  **认知不确定性解释了模型参数的不确定性**。我们并不确定哪种模型权重能够最好地描述数据，但是拥有更多的数据却能降低这种不确定性。这种不确定性在**高风险应用和处理小型稀疏数据**时非常重要

  ![模型可解释性差？你考虑了各种不确定性了吗？](D:\gitrep\ml_theory\ml\mutitask_learning.assets\5b668364a0422.jfif)

  举个例子，假设你想要建立一个能够判断输入图像中的动物是否有可能会吃掉你的模型。然后你的模型只在包含了狮子和长颈鹿的数据集上进行训练，而现在给出一张僵尸的图片作为输入。由于该模型没有学习过僵尸的图片，因此预测结果的不确定性会很高。这种不确定性属于模型的结果，然后如果**你在数据集中给出了更多的僵尸图片，那么模型的不确定性将会降低。**

**数据不确定性**或者称为随机不确定性（**Aleatoric uncertainty**），**指的是观测中固有的噪音。有时事件本身就是随机的，所以在这种情况下，获取更多的数据对我们并没有帮助，因为噪声属于数据固有的。**

为了理解这一点，让我们回到判别食肉动物的模型中。我们的模型可以判断出一张图像中存在狮子，因此会预测出你可能被吃掉。但是，如果狮子现在并不饿呢？这种不确定性就来自于数据。另一个例子则是，有两条看起来一样的蛇，但是其中一条有毒，另一条则没有毒。

随机不确定性可以分为两类：

1. 同方差不确定性（Homoscedastic uncertainty）：这时所有输入具有**相同的不确定性**，数据不存在较为特殊的样例
2. 异方差不确定性（Heteroscedastic uncertainty）：这种不确定性取决于具体的输入数据。例如，对于预测图像中深度信息的模型，毫无特征的平面墙（Featureless wall）将比拥有强消失线（Vanishing lines）的图像具有更高的不确定性。

**测量不确定性（Measurement uncertainty）**：另一个不确定性的来源是测量本身。当测量存在噪声时，不确定性将增加。在上述判别食肉动物的模型中，如果某些图像是通过质量较差的摄像机拍摄的话，那么就有可能会损害模型的置信度。或者在拍摄一只愤怒河马的过程中，由于我们是边跑边拍的，结果导致了成像模糊。

**标签噪声（Noisy labels）**：在监督学习中我们使用标签来训练模型。而如果标签本身带有噪声，那么不确定性也会增加。

不过当前也存在许多方法可以实现对各种类型的不确定性进行建模。这些方法将在本系列的后续文章中进行介绍。现在，假设有一个黑盒模型暴露了自己对预测结果的不确定性，那我们该如果借助这点来调试模型呢？

![在这里插入图片描述](D:\gitrep\ml_theory\ml\mutitask_learning.assets\2019011818571168.png)



#### reference

- [利用不确定性来衡量多任务学习中的损失函数](https://zhuanlan.zhihu.com/p/65137250)





### 6. [A survey on multi-task learning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1707.08114)

![img](D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-3f0b1fb2eaf12571d0576a3428c610ec_1440w.jpg)

与标准的单任务相比，在学习共享表示的同时训练多个任务有两个主要挑战：

- **Loss Function(how to balance tasks)：**多任务学习的损失函数，**对每个任务的损失进行权重分配**，在这个过程中，必须保证所有任务同等重要，而不能让简单任务主导整个训练过程。手动的设置权重是低效而且不是最优的，因此，**自动的学习这些权重或者设计一个对所有权重具有鲁棒性的网络是十分必要和重要的**。
- **Network Architecture(how to share)：**一个高效的多任务网络结构，必须同时兼顾特征共享部分和任务特定部分，既需要学习任务间的泛化表示（避免过拟合），也需要学习每个任务独有的特征（避免欠拟合）。

#### 1. Auxiliary Learning(辅助学习)

除了同时学习多个任务，在有些情况下，我们的关注点只是多任务中的一个或者几个任务的表现。为了更好的理解任务之间的相关性，我们可以通过设置带有各种属性的辅助任务来进行。*辅助任务的目的就是协助我们找到一个更强大，更具有鲁棒性的特征表示，最终让主要任务受益*。关于辅助任务的定义，我们可以根据上文的多任务定义进行延伸，如下表示：

![img](D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-b24605b011170bf4713269b92f516336_1440w.png)

#### 2.Multi-Task Framework Design

**Q：How to properly balance different types of tasks such that training multi-task networks will not be dominated by the easier task(s)?**

分析：第一个问题是，在设计多任务网路过程中，我们如何平衡不同类型的任务，避免在训练过程中，整个网络被简单任务主导，导致任务之间的性能差异巨大。这就涉及到为不同任务的loss function赋上不同的权重，将不同task之间的loss统一成一个损失函数，如果只是简单的将不同任务的loss相加，这样会造成最终模型在有些任务上表现很好，在有的任务上大失水准。背后的原因是不同任务的不同损失函数尺度有很大的差异，因此需要考虑用权值将每个损失函数的尺度统一。

**A：**针对这个问题，最新的解决办法是cvpr2018的一个工作《[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://link.zhihu.com/?target=http%3A//openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)》，这篇文章提出，将不同的loss拉到统一尺度下，这样就容易统一，具体的办法就是利用同方差的不确定性，将不确定性作为噪声，进行训练，详细的讲解可以看我专栏文章：

这里在简单的讲一下同方差的不确定性（Homoscedastic Uncertainty）:属于偶然不确定性，这种不确定性捕捉了不同任务之间的相关性置信度，这种不确定性可以作为不同任务loss赋值的衡量标准。

![img](D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-dac4133d643492318319155746fd7489_1440w.jpg)

其中，Fw是神经网络的输出，在一个有K个任务的模型中，似然估计可以表示为通过概率累乘得到，则极大似然估计可以写成：

![img](D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-db3bc1979fe6fd08c42ae2ca02f1e7d3_1440w.jpg)

其中，σ类似神经网络的参数w，都是可以通过反向传播进行更新，表示的是每个任务输出的置信度。分析上式可知，如果σ增加，相对应的任务loss的权重就会减小，这样就实现了权重的动态规划

**Q：How to build a multi-task learning architecture which is easy to train,parameter-efficient and robust to task weighting?**

分析：如何构建一个统一，易训练，高鲁棒的多任务网络，有多种思想，但是，一个优秀的多任务网络应该具备：(1)特征共享部分和任务特定部分都能自动学习（2）对损失函数权重的选择上更robust

**A：**如下图所示，关于特征共享表示，一般有两种方法，Hard-parameter sharing和soft-parameter-sharing。hard-parameter sharing有一组相同的特征共享层，这种设计大大减少了过拟合的风险；soft-parameter sharing每一个任务都有自己的特征参数和每个子任务之间的约束，这种设计更robust。

![img](D:\gitrep\ml_theory\ml\mutitask_learning.assets\v2-0bf7a60b5e8f4833095d668a611c5ecd_1440w.jpg)

当下最主流的框架都是两种框架的结合，通过结合，能够找到特征共享部分和特定任务部分很好的协调，下面介绍常见的多任务网络的结构设计：

**Fusion Network**是一种通用的特征学习网络，每个任务的上层共享表示是通过学习特定任务的参数，将所有任务的低层特征表示通过线性组合表示出来。代表的网络结构有“Cross-Stitch Network”十字绣网络，了解更多关于该网络，可以去看[论文原文](https://link.zhihu.com/?target=https%3A//www.cv-foundation.org/openaccess/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html)。（论文解读可以看我的专栏文章，[点我跳转](https://zhuanlan.zhihu.com/p/63425561)）

### 7.多任务学习有效的原因

（1）多个相关任务放在一起学习，有相关的部分，但也有不相关的部分。当学习一个任务（Main task）时，与该任务不相关的部分，在学习过程中相当于是噪声，因此，引入噪声可以提高学习的泛化（generalization）效果。

（2)  单任务学习时，梯度的反向传播倾向于陷入局部极小值。多任务学习中不同任务的局部极小值处于不同的位置，通过相互作用，可以帮助隐含层逃离局部极小值。

（3）添加的任务可以改变权值更新的动态特性，可能使网络更适合多任务学习。比如，多任务并行学习，提升了浅层共享层（shared representation）的学习速率，可能，较大的学习速率提升了学习效果。

（4）多个任务在浅层共享表示，可能削弱了网络的能力，降低网络过拟合，提升了泛化效果。

还有很多潜在的解释，为什么多任务并行学习可以提升学习效果（performance）。多任务学习有效，是因为它是建立在多个相关的，具有共享表示（shared representation）的任务基础之上的，因此，需要定义一下，什么样的任务之间是相关的。









## Rerference

- [Multitask-Learning](https://github.com/mbs0221/Multitask-Learning)
- [深度学习（三十四）——深度推荐系统](http://www.jeepxie.net/article/79835.html)
- [从技术角度聊聊，短视频为何让人停不下来？](https://zhuanlan.zhihu.com/p/42777502):smile:
- [利用 TensorFlow 一步一步构建一个多任务学习模型](https://blog.csdn.net/CoderPai/article/details/80087188)
- [An Overview of Multi-Task Learning in Deep Neural Networks](https://ruder.io/multi-task/)
- [Multi-task Learning(Review)多任务学习概述](https://zhuanlan.zhihu.com/p/59413549):confounded:
- [极大似然估计](https://blog.csdn.net/zengxiantao1994/article/details/72787849)

**推荐排序**

- [知乎推荐页Ranking经验分享](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247487240&idx=1&sn=f38e4a5cd73a2d2f1d52ee5eda418528&chksm=fbd4bd64cca334720629ba997aca98642bc1e1b25c422623ddf07e30cb9a5f5068cf22af0048&scene=21#wechat_redirect)
- [机器学习爱好者](https://zhuanlan.zhihu.com/fengdu78)

