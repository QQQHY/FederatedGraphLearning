<!--Title-->
# Federated Graph Classification over Non-IID Graphs

## Basic Message
  - Author: Han Xie, Jing Ma, Li Xiong, Carl Yang
  - Publication: NeurIPS 2021
  - Date: 25 Jun 2021
  - Link: <https://arxiv.org/abs/2106.13423>



## Take Home Message

图上的non-IID 
non-IID 有什么问题
聚类何以解决
为什么聚合根据梯度而不是模型参数
<!-- 
take home message 总结文章的核心思想
写完笔记之后最后填，概述文章的内容，也是查阅笔记的时候先看的一段。
写文章summary切记需要通过自己的思考，用自己的语言描述。 
-->
  
## Abstract

Abstract: Federated learning has emerged as an important paradigm for training machine learning models in different domains. For graph-level tasks such as graph classification, graphs can also be regarded as a special type of data samples, which can be collected and stored in separate local systems. Similar to other domains, multiple local systems, each holding a small set of graphs, may benefit from collaboratively training a powerful graph mining model, such as the popular graph neural networks (GNNs). To provide more motivation towards such endeavors, we analyze real-world graphs from different domains to confirm that they indeed share certain graph properties that are statistically significant compared with random graphs. However, we also find that different sets of graphs, even from the same domain or same dataset, are non-IID regarding both graph structures and node features. To handle this, we propose a graph clustered federated learning (GCFL) framework that dynamically finds clusters of local systems based on the gradients of GNNs, and theoretically justify that such clusters can reduce the structure and feature heterogeneity among graphs owned by the local systems. Moreover, we observe the gradients of GNNs to be rather fluctuating in GCFL which impedes high-quality clustering, and design a gradient sequence-based clustering mechanism based on dynamic time warping (GCFL+). Extensive experimental results and in-depth analysis demonstrate the effectiveness of our proposed frameworks.

<!-- 
背景
问题
现状
缺陷
方法
结果 -->

- 联邦学习共享图性质
- non-IID

## Introduction
<!-- 背景知识 -->

### Related works

<!-- 哪些文章的哪些结论，与本文联系 -->
#### 联邦学习数据异质性,非独立同分布

FedAvg[27]依赖于SGD优化，其收敛性是建立在数据IID的假设上的，[53,25,18]实验证明了收敛会因为non-IID而减缓且不平稳，准确率会下降。  
[53,15,12]的策略需要数据的有限共享.  
对于non-IID环境下的收敛性保证，分别通过假设有界梯度[39,51]或附加噪声[19].  
[26,18,25]从减少variance的角度着手.  
进一步的工作探索了模型无关元学习与个性化学习的联系，在学习泛化全局模型后进行局部微调[8,4].  
[6,24] 解耦局部与全局优化，但每个client保有自己的模型会带来很高的通讯成本.  
个性化学习是解决思路之一，而聚类可以看做簇级别的个性化.  

[4] **personalization** Fei Chen, Mi Luo, Zhenhua Dong, Zhenguo Li, and Xiuqiang He. Federated meta-learning with fast convergence and efficient communication. arXiv preprint arXiv:1802.07876, 2018.  
[6] **personalization** Canh T. Dinh, Nguyen H. Tran, and Tuan Dung Nguyen. Personalized federated learning with moreau envelopes. In NeurIPS, 2021.  
[8] **personalization** Alireza Fallah, Aryan Mokhtari, and Asuman Ozdaglar. Personalized federated learning: A meta-learning approach. In NeurIPS, 2020.  
[12] **data sharing** Li Huang, Yifeng Yin, Zeng Fu, Shifa Zhang, Hao Deng, and Dianbo Liu. Loadaboost: Loss- based adaboost federated machine learning on medical data. PLoS ONE, 15(4):e0230706, 2020.  
[15] **data sharing** Eunjeong Jeong, Seungeun Oh, Hyesung Kim, Jihong Park, Mehdi Bennis, and Seong-Lyun Kim. Communication-efficient on-device machine learning: Federated distillation and augmen- tation under non-iid private data. In NIPSW, 2018.  
[18] **non-IID Exp, variance** Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J Reddi, Sebastian U Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In ICML, 2019.  
[19] **additional noise** Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. Tighter theory for local sgd on identical and heterogeneous data. In AISTATS, 2020.
[24] **personalization** Tian Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith. Ditto: Fair and robust federated learning through personalization. In ICML, 2021.  
[25] **non-IID Exp, variance** Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. In Proceedings ofMachine Learning and Systems, 2020.  
[26] **variance** Xianfeng Liang, Shuheng Shen, Jingchang Liu, Zhen Pan, Enhong Chen, and Yifei Cheng. Vari- ance reduced local sgd with lower communication complexity. arXiv preprint arXiv:1912.12844, 2019.
[27] **FedAvg** Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial Intelligence and Statistics, 2017.  
[33] **Cluster FL** Felix Sattler, Klaus-Robert Müller, and Wojciech Samek. Clustered federated learning: Model- agnostic distributed multitask optimization under privacy constraints. TNNLS, pages 1–13, 2020.  
[53] **non-IID Exp, data sharing** Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, and Vikas Chandra. Federated learning with non-iid data. arXiv preprint arXiv:1806.00582, 2018.  

#### 联邦图学习

不同于欧式数据，联邦学习在图数据上还未得到广泛研究。  
[22]Anusha Lalitha, Osman Cihan Kilinc, Tara Javidi, and Farinaz Koushanfar. Peer-to-peer federated learning on graphs. arXiv preprint arXiv:1901.11173, 2019.   
[22]引入了联邦图学习（节点级别）   
[3] Debora Caldarola, Massimiliano Mancini, Fabio Galasso, Marco Ciccone, Emanuele Rodolà, and Barbara Caputo. Cluster-driven graph federated learning over multiple domains. In CVPRW, 2021.     
[3]研究跨域异构性问题，利用图卷积网络(GCNs)建模域之间的交互  
[29] Chuizheng Meng, Sirisha Rambhatla, and Yan Liu. Cross-node federated graph neural network for spatio-temporal data modeling. In KDD, 2021.  
[29]研究联邦时空数据建模，通过利用基于GNN的模型来捕捉client之间的空间关系。  
[5] Mingyang Chen, Wen Zhang, Zonggang Yuan, Yantao Jia, and Huajun Chen. Fede: Embedding knowledge graphs in federated setting. arXiv preprint arXiv:2010.12882, 2020.  
[5]提出了通用联邦知识图嵌入框架。  
[16] Meng Jiang, Taeho Jung, Ryan Karl, and Tong Zhao. Federated dynamic gnn with secure aggregation. arXiv preprint arXiv:2009.07351, 2020.  
[54] Jun Zhou, Chaochao Chen, Longfei Zheng, Huiwen Wu, Jia Wu, Xiaolin Zheng, Bingzhe Wu, Ziqi Liu, and Li Wang. Vertically federated graph neural network for privacy-preserving node classification. arXiv preprint arXiv:2005.11903, 2020.  
[40] Chuhan Wu, Fangzhao Wu, Yang Cao, Yongfeng Huang, and Xing Xie. Fedgnn: Federated graph neural network for privacy-preserving recommendation. arXiv preprint arXiv:2102.04925, 2021.    
[16,54,40]关注FL-GNN的隐私问题。  
[37] Binghui Wang, Ang Li, Hai Li, and Yiran Chen. Graphfl: A federated learning framework for semi-supervised node classification on graphs. arXiv preprint arXiv:2012.04187, 2020.   
[37]将模型无关元学习(MAML)引入到图FL中，在处理non-IID图数据的同时保持模型的通用性  
[52] Ke Zhang, Carl Yang, Xiaoxiao Li, Lichao Sun, and Siu Ming Yiu. Subgraph federated learning with missing neighbor generation. In NeurIPS, 2021.  
[52]研究了子图FL设置中的缺失邻居生成问题  
[38] Chunnan Wang, Bozhou Chen, Geng Li, and Hongzhi Wang. Fl-agcns: Federated learning framework for automatic graph convolutional network search. In ICML, 2021.      
[38]提出了一种FL-GCN体系结构搜索方法   
[11] Chaoyang He, Keshav Balasubramanian, Emir Ceyani, Carl Yang, Han Xie, Lichao Sun, Lifang
He, Liangwei Yang, Philip S Yu, Yu Rong, Peilin Zhao, Junzhou Huang, Murali Annavaram, and Salman Avestimehr. Fedgraphnn: A federated learning system and benchmark for graph neural networks. arXiv preprint arXiv:2104.07145, 2021.  
[11]实现了一个FL-GNN基准     
现有的大多数工作都考虑到图的节点分类和链接预测(节点、子图级别)，不能简单地应用到我们的图分类(图级别)设置中。  
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

- [论文笔记：NIPS 2021 Federated Graph Classification over Non-IID Graphs (GCFL)](https://zhuanlan.zhihu.com/p/430623053)
- [NeurIPS'21 | Non-IID图数据上的联邦图学习](https://mp.weixin.qq.com/s?__biz=Mzg5MjY0NTQ1MQ==&mid=2247484504&idx=1&sn=9be593fb9e0f33e543e7a207ecd28872&chksm=c03bbd7cf74c346a2a23b7c117c598633295cce75f0b03d35252f1597f051f5f9600534d78d1&token=2123268658&lang=zh_CN#rd)
- [NIPS OpenReview](https://openreview.net/forum?id=yJqcM36Qvnu)
- [code](https://github.com/Oxfordblue7/GCFL)
  
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
