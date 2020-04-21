# 深度学习与推荐系统

## 问题记录：

- 消除用户和物品的打分 p26

- Neural CF GMF p64

- DCN  tf.tensordot p62

- PNN  内积 外积p69

- 记忆能力、泛化能力 p71 假如一个覆盖率只有1%的强特征会背记忆下来吗比如搜索过车系或者下线索 

- Deep Cross  cross 的理解  cross 每次计算就会得出类似于乘积，相当于做了一次特征交叉，然后又经过w进行加权求和，然后每个维度上都有$x_{0}$ 的信息，下次再做交叉的时候就相当于一个高阶 p76

- Embedding 层收敛速度慢 p78

  embedding 每次只有非0特征相连的embedding 才会被训练，那覆盖率比较低的特征是不是要过采样一下。或者是优化adam方式上进行修正
  
- [NFM]( https://github.com/xxxmin/ctr_Keras) 代码实现

- 数据过大需要用tfrecord和多线程队列减少io时间

- Item2vec 摒弃时间窗口有以下几问题

  1. 时间窗口的局部性  对推荐效果的影响，与推荐的具体场景应当相关，假如每个item seq 都是代表用户短期的用户偏好，则两者差异较小否则两者差异相对较大，item2vec 跟关注于全局性，word2vec 更关注于局部


- Node2vec 的同质性和结构性的理解 p115 

- [estimator load pretrain word embedding ](https://androidkt.com/pre-trained-word-embedding-tensorflow-using-estimator-api/)

- 局部敏感哈希 固定间隔的超平面将向量空间分割成不同的哈希桶  p125

- EGES 基于知识图谱训练完embedding 然后把基于行为或者其他信息进行训练会不会导致于word2vec 得出相似的embedding ,先验知识是固定的 

  

  

  
  
  
  
  
  
  
  
  