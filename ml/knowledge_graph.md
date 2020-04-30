# 知识图谱

## 1. 知识图谱概念与技术 笔记

1. 知识图谱的数值表示  基于距离 SE TransE TransH TransR TransD p49

2. 概率图模型 马尔可夫链 p61 :confused:

3. 命名实体识别是识别句子的词序列中具有特定意义的实体，并将其标注为人名、机构名、日期、地点、时间和职位等类别任务；共指消解识别句子中同一实体的所有不同表达 p76

4. 负采样 p77

5. 领域短语挖掘

   短语质量的评估：频率、一致性、信息量（机器学习和这篇文章）、完整性 p87

6. 领域短语挖掘方法 p89

     无监督。**频繁模式挖掘**:open_mouth:

   ```mermaid
   graph LR 
   A[语料]-->B(候选短语生成)-->C(统计特征计算)-->D(质量评分)--> E(短语)
   ```

   统计特征：TF-IDF、PMI、左(右)邻字熵、C-value,NC-value

   远程监督
   
7. 同义词挖掘  p96

     - 基于同义词资源的方法
     - 基于模式匹配的方法
     - 自举法
     - 图模型挖掘   模块度 :confused:

8.  缩略词 p103
  

缩略词的检测与抽取以模式匹配为主，抽取结果的清洗和筛选。

字符匹配程度，词性特征

  缩略词的预测

  - 基于规则
  
  - **条件随机场CRF** :confused: p107
  
    序列标注
  
  - RNN 

  在中文相关的处理中，通常要将字符级别向量表示和词汇级别向量表示等不同粒度的语言信息输入到深度神经网络模型中，才能取得较好的效果。

9. **命名实体识别(Named Entity Recongnition)** p109
  
  NER 的输入是一个句子对应的单词序列$s=<w_{1},w_{2}...w_{N}>$ ,输出是一个三元组集合，其中每个三元组的形式为$(I_{s},I_{e},t)$  分别表示命名实体再s中的开始和结束位置，而t是实体类型。
  
  粗粒度命名实体识别(Coarse-grained Entity Typing) 每个命名实体只分配一个类型
  
  细粒度命名实体识别(Fine-grained Entity Typing)
  
  
  
  传统的NER 方法主要分为三类：基于规则、词典和在线知识库，监督学习方法和半监督学习方法
  
  **监督学习方法** p112
  
  当应用监督学习时，NER被建模为序列标注问题(**Sequence labeling**)。NER任务使用BIO 标注法。BIO标注法是NER任务常用的标注法，其中B表示实体的起始位置，I 表示中间或结束位置，O表示相应的字符是不是实体。
  
  **基于序列标注的建模接收文本为输入，产生相应的BIO标注输出。**
  
  - HMM
  
  - CRF
  
    NER 特征工程 核心词特征/词典特征/构词特征/词形特征/词缀特征/词性特征
  
  **半监督学习**
  
  
  
  **基于深度学习的NER 方法**
  
  典型的NER 框架包括：输入embedding、上下问编码(Context Encoder)、标签解码器(Tag Decoder)
  
  输入embedding: 词/字，通常使用CNN RNN 模型提取字向量
  
  Survey p121
  
10. 关系抽取

关系抽取(**Realtion Extraction**)的结果是关系实例，构成了知识图谱中的边。一般而言，关系抽取的结果是三元组**<主体(Subject),谓词(Predicate),客体(Object)>​**,表示主体和客体之间存在的谓词所表达的关系。

11. 概念知识图谱

概念认知是对某个形态的数据输入产生符号话概念输出的过程。

isA:instanceOf subclassOf 

应用：p166 

**实例化**

**概念化**

**isA 关系抽取**

isA关系的抽取是构建概念图谱的核心。isA关系抽取的方法分三种：基于模式(Pattern)、基于在线百科的方法以及基于词向量(word embedding)的方法。

词汇概念体系中的基本关系是词汇之家的**上下位关系**。比如，"apple isA fruit",apple是fruit的下位词，fruit是apple的上位词。

**isA 关系补全两种典型的补全思路：利用isA的传递性，相似实体(协同过滤思想)** p177

**Noisy-Or​**:confounded:

**isA 关系纠错**

知识图谱的关系存储：

- 基于三列表  
- 基于属性
- 基于垂直表 spark S2RDF
- 基于全索引
- 基于图 邻接表(hadoop SHARD)，邻接矩阵

知识图谱的查询和搜索

SPARQL ->RDF

  


​     


## reference
- [Modularity的计算方法——社团检测中模块度计算公式详解](http://www.yalewoo.com/modularity_community_detection.html)

- 


​     

​     

​     

