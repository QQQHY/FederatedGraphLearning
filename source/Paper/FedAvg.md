<!--Title-->
# Communication-Efficient Learning of Deep Networksfrom Decentralized Data

## Basic Message

- Author: H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas
- Publication: AISTATS 2017
- Date: 17 Feb 2016
- Link: <https://arxiv.org/abs/1602.05629>

## Take Home Message

<!-- 
take home message 总结文章的核心思想
写完笔记之后最后填，概述文章的内容，也是查阅笔记的时候先看的一段。
写文章summary切记需要通过自己的思考，用自己的语言描述。 
-->
  
## Abstract

Modern mobile devices have access to a wealth of data suitable for learning models, which in turn can greatly improve the user experience on the device. For example, language models can improve speech recognition and text entry, and image models can automatically select good photos. However, this rich data is often privacy sensitive, large in quantity, or both, which may preclude logging to the data center and training there using conventional approaches. We advocate an alternative that leaves the training data distributed on the mobile devices, and learns a shared model by aggregating locally-computed updates. We term this decentralized approach Federated Learning.
We present a practical method for the federated learning of deep networks based on iterative model averaging, and conduct an extensive empirical evaluation, considering five different model architectures and four datasets. These experiments demonstrate the approach is robust to the unbalanced and non-IID data distributions that are a defining characteristic of this setting. Communication costs are the principal constraint, and we show a reduction in required communication rounds by 10-100x as compared to synchronized stochastic gradient descent.

<!-- 
背景
问题
现状
缺陷
方法
结果 -->



## Introduction
<!-- 背景知识 -->
- 和传统的分布式优化相比，联邦学习的关键属性：
  - Non-IID：、客户端数据非独立同分布
  - Unbalanced：客户端数据量差异很大
  - Massively distributed：客户端数量极其多
  - Limited communication：通讯情况不同且通信质量难以保证

### Related works

<!-- 哪些文章的哪些结论，与本文联系 -->

### Problem Statement
<!-- 问题陈述 -->

<!-- - 需要解决的问题是什么？
- 扩充知识面
  - 重建别人的想法，通过读Introduction思考别人是如何想出来的
- 假设
  - 有什么基本假设、是否正确、假设是否可以系统化验证
  - 假设很有可能是错的，还可以用哪些其他方法来验证
- 应用场景 -->

## Methods
<!-- 文章设计的方法 -->

<!--
解决问题的方法/算法是什么
	主要理论、主要公式、主要创意
	创意的好处、成立条件
	为什么要用这种方法
	是否基于前人的方法？
有什么缺点、空缺、漏洞、局限
	效果不够好
	考虑不顾全面
	在应用上有哪些坏处，怎么引起的
还可以用什么方法？
方法可以还用在哪？有什么可以借鉴的地方？ 
-->
  
## Evaluation & Experiments
<!-- 实验评估 -->

<!-- 
- 作者如何评估自己的方法
- 实验的setup
   § 数据集
    □ 名称、基本参数、异同，为什么选择（Baseline）
    □ 如何处理数据以便于实验
   § 模型
   § baseline
   § 与什么方法比较
- 实验证明了哪些结论
- 实验有什么可借鉴的
- 实验有什么不足 
-->

## Discuss & Conclusion

<!-- 
作者给了哪些结论
- 哪些是strong conclusions, 哪些又是weak的conclusions?
- 文章的讨论、结论部分，
   § 结尾的地方往往有些启发性的讨论 
-->

## Reference
<!-- 列出相关性高的参考文献-->
  
## Useful link

<!-- 
论文笔记、讲解
Code Slides Web Review
Author Page 
-->

## Notes

<!-- - 不符合此框架，但需要额外记录的笔记。
- 英语单词、好的句子 -->

 <!-- 
读文章步骤：
  迭代式读法
  先读标题、摘要 图表 再读介绍 读讨论 读结果 读实验
  通读全文，能不查字典最好先不查字典
  边读边总结，总结主要含义，注重逻辑推理
  
 摘要
  多数文章看摘要，少数文章看全文

 实验
  结合图表

 理论：
  有什么样的假设 是否合理 其他设定
  推导是否完善 用了什么数学工具
  
 idea来源：
  突出理论还是实践
   理论：数学
   实践：跑通code，调参过程中改进，找到work的方案后思考成因
  针对特定缺点，设计方案 
-->
