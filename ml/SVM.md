# SVM

##  核函数的定义和作用

其实是一个非常简单的概念。

首先给你两个向量 ![[公式]](SVM.assets/equation.svg) 。在一般的机器学习方法，比如 SVM 里面，这里一个向量是一个实体。比如一个向量代表一个人。每个向量有两个维度，身高和体重。比如可以有$x$

现在要求两个人的相似度，最简单的方法是计算它们的内积 ![[公式]](https://www.zhihu.com/equation?tex=%3C%5Ctextbf+x%2C+%5Ctextbf+z%3E) 。这很简单，只要按照维度相乘求和就可以了。

![[公式]](https://www.zhihu.com/equation?tex=%3C%5Ctextbf+x%2C+%5Ctextbf+z%3E+%3D+180+%2A+160+%2B+70+%2A+50+%3D+32300) 

但是有的时候（比如 SVM 的数据线性不可分的时候），我们可能会想对数据做一些操作。我们可能认为体重的二次方，身高的二次方，或者身高体重的乘积是更重要的特征，我们把这个操作记为过程 ![[公式]](SVM.assets/equation.svg) ，比如可能有$x$

我们认为 $x$ 比 $x$ 更能表示一个人的特征。我们再计算两个人的相似度时，使用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%28%5Ctextbf+x%29) 与 ![[公式]](SVM.assets/equation.svg) 的内积：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cbegin%7Bsplit%7D+%3C%5Cphi+%28%5Ctextbf+x%29%2C+%5Cphi%28%5Ctextbf+z%29%3E+%26%3D+%3C%28180%5E2%2C70%5E2%2C%5Csqrt+2%2A180%2A70%29%2C%28160%5E2%2C50%5E2%2C%5Csqrt+2%2A160%2A50%29%3E%5C%5C+%26%3D180%5E2%2A160%5E2%2B70%5E2%2A50%5E2%2B%5Csqrt+2%2A180%2A70%2A%5Csqrt+2%2A160%2A50%5C%5C+%26%3D1043290000+%5Cend%7Bsplit%7D+%5Cend%7Bequation%7D) 

在上面的操作中，我们总共要计算 **11 次乘法，2 次加法**。

但是如果我们定义核函数

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cbegin%7Bsplit%7D+K%28%5Ctextbf+x%2C%5Ctextbf+z%29%26%3D%28%3C%5Ctextbf+x%2C+%5Ctextbf+z%3E%29%5E2%5C%5C+%5Cend%7Bsplit%7D+%5Cend%7Bequation%7D) 

那么有

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cbegin%7Bsplit%7D+K%28%5Ctextbf+x%2C%5Ctextbf+z%29%26%3D%28%3C%5Ctextbf+x%2C+%5Ctextbf+z%3E%29%5E2%5C%5C+%26%3D%28x_1z_1%2Bx_2z_2%29%5E2%5C%5C+%26%3D%28180%2A160%2B70%2A50%29%5E2%5C%5C+%26%3D1043290000+%5Cend%7Bsplit%7D+%5Cend%7Bequation%7D) 

可以看到 ![[公式]](https://www.zhihu.com/equation?tex=K%28%5Ctextbf+x%2C%5Ctextbf+z%29%3D%3C%5Cphi%28%5Ctextbf+x%29%2C%5Cphi%28%5Ctextbf+z%29%3E) 。但是这次我们只计算了 **3 次乘法，1 次加法**。

------

所以其实核函数就是这么一回事：

当我们需要先对数据做转换，然后求内积的时候，这样的一系列操作往往成本过高（有时候根本不可能，因为我们可能想要升到无穷维）。因此我们可以直接定义一个核函数 K 直接求出做转换后求内积的结果，从而降低运算量。



- [支持向量机（SVM）——原理篇](https://zhuanlan.zhihu.com/p/31886934)
- [核函数](https://www.zhihu.com/question/24627666/answer/1085861632)

