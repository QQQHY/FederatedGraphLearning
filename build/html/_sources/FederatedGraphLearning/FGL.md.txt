# Federated Graph Learning

Generally sorted by category and date.

***

## FL & GNN

- **Vertically Federated Graph Neural Network for Privacy-Preserving Node Classification**
  - Author: Jun Zhou, Chaochao Chen, Longfei Zheng, Huiwen Wu, Jia Wu, Xiaolin Zheng, Bingzhe Wu, Ziqi Liu, Li Wang
  - Publication: Preprint
  - Date: 25 May 2020
  - Link: <https://arxiv.org/pdf/2005.11903>
  - Abstract: Recently, Graph Neural Network (GNN) has achieved remarkable progresses in various real-world tasks on graph data, consisting of node features and the adjacent information between different nodes. High-performance GNN models always depend on both rich features and complete edge information in graph. However, such information could possibly be isolated by different data holders in practice, which is the so-called data isolation problem. To solve this problem, in this paper, we propose VFGNN, a federated GNN learning paradigm for privacy-preserving node classification task under data vertically partitioned setting, which can be generalized to existing GNN models. Specifically, we split the computation graph into two parts. We leave the private data (i.e., features, edges, and labels) related computations on data holders, and delegate the rest of computations to a semi-honest server. We also propose to apply differential privacy to prevent potential information leakage from the server. We conduct experiments on three benchmarks and the results demonstrate the effectiveness of VFGNN.
  - Privacy-Preserving Graph Neural Network for Node Classification
- **Federated Dynamic GNN with Secure Aggregation.**
  - Author: Meng Jiang, Taeho Jung, Ryan Karl, Tong Zhao
  - Publication: Preprint
  - Date: 15 Sep 2020
  - Link: <https://arxiv.org/pdf/2009.07351>
  - Abstract: Given video data from multiple personal devices or street cameras, can we exploit the structural and dynamic information to learn dynamic representation of objects for applications such as distributed surveillance, without storing data at a central server that leads to a violation of user privacy? In this work, we introduce Federated Dynamic Graph Neural Network (Feddy), a distributed and secured framework to learn the object representations from multi-user graph sequences: i) It aggregates structural information from nearby objects in the current graph as well as dynamic information from those in the previous graph. It uses a self-supervised loss of predicting the trajectories of objects. ii) It is trained in a federated learning manner. The centrally located server sends the model to user devices. Local models on the respective user devices learn and periodically send their learning to the central server without ever exposing the user's data to server. iii) Studies showed that the aggregated parameters could be inspected though decrypted when broadcast to clients for model synchronizing, after the server performed a weighted average. We design an appropriate aggregation mechanism of secure aggregation primitives that can protect the security and privacy in federated learning with scalability. Experiments on four video camera datasets (in four different scenes) as well as simulation demonstrate that Feddy achieves great effectiveness and security.
- **FedGNN: Federated Graph Neural Network for Privacy-Preserving Recommendation**
  - Author: Chuhan Wu, Fangzhao Wu, Yang Cao, Yongfeng Huang, Xing Xie
  - Publication: KDD 2021
  - Date: 9 Feb 2021
  - Link: <https://arxiv.org/pdf/2102.04925>  
  - Abstract: Graphs have been widely used in data mining and machine learning due to their unique representation of real-world objects and their interactions. As graphs are getting bigger and bigger nowadays, it is common to see their subgraphs separately collected and stored in multiple local systems. Therefore, it is natural to consider the subgraph federated learning setting, where each local system holds a small subgraph that may be biased from the distribution of the whole graph. Hence, the subgraph federated learning aims to collaboratively train a powerful and generalizable graph mining model without directly sharing their graph data. In this work, towards the novel yet realistic setting of subgraph federated learning, we propose two major techniques: (1) FedSage, which trains a GraphSage model based on FedAvg to integrate node features, link structures, and task labels on multiple local subgraphs; (2) FedSage+, which trains a missing neighbor generator along FedSage to deal with missing links across local subgraphs. Empirical results on four real-world graph datasets with synthesized subgraph federated learning settings demonstrate the effectiveness and efficiency of our proposed techniques. At the same time, consistent theoretical implications are made towards their generalization ability on the global graphs.
- **FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks.**
  - Author: Chaoyang He, Keshav Balasubramanian, Emir Ceyani, Carl Yang, Han Xie, Lichao Sun, Lifang He, Liangwei Yang, Philip S. Yu, Yu Rong, Peilin Zhao, Junzhou Huang, Murali Annavaram, Salman Avestimehr
  - Publication: ICLR-DPML 2021 & MLSys21-GNNSys 2021
  - Date: 14 Apr 2021
  - Link: <https://arxiv.org/pdf/2104.07145>
  - Abstract: Graph Neural Network (GNN) research is rapidly growing thanks to the capacity of GNNs in learning distributed representations from graph-structured data. However, centralizing a massive amount of real-world graph data for GNN training is prohibitive due to privacy concerns, regulation restrictions, and commercial competitions. Federated learning (FL), a trending distributed learning paradigm, provides possibilities to solve this challenge while preserving data privacy. Despite recent advances in vision and language domains, there is no suitable platform for the FL of GNNs. To this end, we introduce FedGraphNN, an open FL benchmark system that can facilitate research on federated GNNs. FedGraphNN is built on a unified formulation of graph FL and contains a wide range of datasets from different domains, popular GNN models, and FL algorithms, with secure and efficient system support. Particularly for the datasets, we collect, preprocess, and partition 36 datasets from 7 domains, including both publicly available ones and specifically obtained ones such as hERG and Tencent. Our empirical analysis showcases the utility of our benchmark system, while exposing significant challenges in graph FL: federated GNNs perform worse in most datasets with a non-IID split than centralized GNNs; the GNN model that attains the best result in the centralized setting may not maintain its advantage in the FL setting. These results imply that more research efforts are needed to unravel the mystery behind federated GNNs. Moreover, our system performance analysis demonstrates that the FedGraphNN system is computationally efficient and secure to large-scale graphs datasets. We maintain the source code at this https URL.
  - *Github* [FedGraphNN](https://github.com/FedML-AI/FedGraphNN)
- **Federated Graph Classification over Non-IID Graphs**
  - Author: Han Xie, Jing Ma, Li Xiong, Carl Yang
  - Publication: NeurIPS 2021
  - Date: 25 Jun 2021
  - Link: <https://arxiv.org/abs/2106.13423>
  - Abstract: Federated learning has emerged as an important paradigm for training machine learning models in different domains. For graph-level tasks such as graph classification, graphs can also be regarded as a special type of data samples, which can be collected and stored in separate local systems. Similar to other domains, multiple local systems, each holding a small set of graphs, may benefit from collaboratively training a powerful graph mining model, such as the popular graph neural networks (GNNs). To provide more motivation towards such endeavors, we analyze real-world graphs from different domains to confirm that they indeed share certain graph properties that are statistically significant compared with random graphs. However, we also find that different sets of graphs, even from the same domain or same dataset, are non-IID regarding both graph structures and node features. To handle this, we propose a graph clustered federated learning (GCFL) framework that dynamically finds clusters of local systems based on the gradients of GNNs, and theoretically justify that such clusters can reduce the structure and feature heterogeneity among graphs owned by the local systems. Moreover, we observe the gradients of GNNs to be rather fluctuating in GCFL which impedes high-quality clustering, and design a gradient sequence-based clustering mechanism based on dynamic time warping (GCFL+). Extensive experimental results and in-depth analysis demonstrate the effectiveness of our proposed frameworks.
- **Subgraph Federated Learning with Missing Neighbor Generation**
  - Author: Ke Zhang, Carl Yang, Xiaoxiao Li, Lichao Sun, Siu Ming Yiu
  - Publication: NeurIPS 2021
  - Date: 25 Jun 2021
  - Link: <https://arxiv.org/abs/2106.13430>  
  - Abstract:
    Graphs have been widely used in data mining and machine learning due to their unique representation of real-world objects and their interactions. As graphs are getting bigger and bigger nowadays, it is common to see their subgraphs separately collected and stored in multiple local systems. Therefore, it is natural to consider the subgraph federated learning setting, where each local system holds a small subgraph that may be biased from the distribution of the whole graph. Hence, the subgraph federated learning aims to collaboratively train a powerful and generalizable graph mining model without directly sharing their graph data. In this work, towards the novel yet realistic setting of subgraph federated learning, we propose two major techniques: (1) FedSage, which trains a GraphSage model based on FedAvg to integrate node features, link structures, and task labels on multiple local subgraphs; (2) FedSage+, which trains a missing neighbor generator along FedSage to deal with missing links across local subgraphs. Empirical results on four real-world graph datasets with synthesized subgraph federated learning settings demonstrate the effectiveness and efficiency of our proposed techniques. At the same time, consistent theoretical implications are made towards their generalization ability on the global graphs.
  - Note
    - FedSage : FedAvg + GraphSage
    - FedSage+ : FedSage + Missing neighbor generator
    - [Meeting_20211028_FedSage](https://docs.google.com/presentation/d/1tC69KdqVy2yEOD6Ev-hBd0LiloJPvUemqAl5hFIvz5s/edit?usp=sharing)

<!--
- ****
  - Author: 
  - Publication: 
  - Date: 
  - Link: <>
  - Abstract:
-->

- **ASFGNN: Automated Separated-Federated Graph Neural Network**
  - Author: Longfei Zheng, Jun Zhou, Chaochao Chen, Bingzhe Wu, Li Wang, Benyu Zhang
  - Publication: preprint
  - Date: 6 Nov 2020
  - Link: <https://arxiv.org/pdf/2011.03248>
  - Abstract: Graph Neural Networks (GNNs) have achieved remarkable performance by taking advantage of graph data. The success of GNN models always depends on rich features and adjacent relationships. However, in practice, such data are usually isolated by different data owners (clients) and thus are likely to be Non-Independent and Identically Distributed (Non-IID). Meanwhile, considering the limited network status of data owners, hyper-parameters optimization for collaborative learning approaches is time-consuming in data isolation scenarios. To address these problems, we propose an Automated Separated-Federated Graph Neural Network (ASFGNN) learning paradigm. ASFGNN consists of two main components, i.e., the training of GNN and the tuning of hyper-parameters. Specifically, to solve the data Non-IID problem, we first propose a separated-federated GNN learning model, which decouples the training of GNN into two parts: the message passing part that is done by clients separately, and the loss computing part that is learnt by clients federally. To handle the time-consuming parameter tuning problem, we leverage Bayesian optimization technique to automatically tune the hyper-parameters of all the clients. We conduct experiments on benchmark datasets and the results demonstrate that ASFGNN significantly outperforms the naive federated GNN, in terms of both accuracy and parameter-tuning efficiency.
  
- **GraphFL: A Federated Learning Framework for Semi-Supervised Node Classification on Graphs**
  - Author: Binghui Wang, Ang Li, Hai Li, Yiran Chen
  - Publication: preprint
  - Date: 8 Dec 2020
  - Link: <https://arxiv.org/pdf/2012.04187>
  - Abstract: Graph-based semi-supervised node classification (GraphSSC) has wide applications, ranging from networking and security to data mining and machine learning, etc. However, existing centralized GraphSSC methods are impractical to solve many real-world graph-based problems, as collecting the entire graph and labeling a reasonable number of labels is time-consuming and costly, and data privacy may be also violated. Federated learning (FL) is an emerging learning paradigm that enables collaborative learning among multiple clients, which can mitigate the issue of label scarcity and protect data privacy as well. Therefore, performing GraphSSC under the FL setting is a promising solution to solve real-world graph-based problems. However, existing FL methods 1) perform poorly when data across clients are non-IID, 2) cannot handle data with new label domains, and 3) cannot leverage unlabeled data, while all these issues naturally happen in real-world graph-based problems. To address the above issues, we propose the first FL framework, namely GraphFL, for semi-supervised node classification on graphs. Our framework is motivated by meta-learning methods. Specifically, we propose two GraphFL methods to respectively address the non-IID issue in graph data and handle the tasks with new label domains. Furthermore, we design a self-training method to leverage unlabeled graph data. We adopt representative graph neural networks as GraphSSC methods and evaluate GraphFL on multiple graph datasets. Experimental results demonstrate that GraphFL significantly outperforms the compared FL baseline and GraphFL with self-training can obtain better performance.
  -
- **FL-AGCNS: Federated Learning Framework for Automatic Graph Convolutional Network Search**
  - Author: Chunnan Wang, Bozhou Chen, Geng Li, Hongzhi Wang
  - Publication: preprint
  - Date: 9 Apr 2021
  - Link: <https://arxiv.org/pdf/2104.04141>
  - Abstract: Recently, some Neural Architecture Search (NAS) techniques are proposed for the automatic design of Graph Convolutional Network (GCN) architectures. They bring great convenience to the use of GCN, but could hardly apply to the Federated Learning (FL) scenarios with distributed and private datasets, which limit their applications. Moreover, they need to train many candidate GCN models from scratch, which is inefficient for FL. To address these challenges, we propose FL-AGCNS, an efficient GCN NAS algorithm suitable for FL scenarios. FL-AGCNS designs a federated evolutionary optimization strategy to enable distributed agents to cooperatively design powerful GCN models while keeping personal information on local devices. Besides, it applies the GCN SuperNet and a weight sharing strategy to speed up the evaluation of GCN models. Experimental results show that FL-AGCNS can find better GCN models in short time under the FL framework, surpassing the state-of-the-arts NAS methods and GCN models.

- **Cluster-driven Graph Federated Learning over Multiple Domains**
  - Author:
  - Publication:
  - Date:
  - Link: <https://arxiv.org/pdf/2104.14628>
  - Abstract:
  -
- **FedGL: Federated Graph Learning Framework with Global Self-Supervision**
  - Author:
  - Publication:
  - Date:
  - Link: <https://arxiv.org/pdf/2105.03170>
  - Abstract:
- **Federated Graph Learning -- A Position Paper**
  - Author:
  - Publication:
  - Date:
  - Link: <https://arxiv.org/pdf/2105.11099>
  - Abstract:
- **SpreadGNN: Serverless Multi-task Federated Learning for Graph Neural Networks**
  - Author:
  - Publication:
  - Date:
  - Link: <https://arxiv.org/pdf/2106.02743>
  - Abstract:
- **Cross-Node Federated Graph Neural Network for Spatio-Temporal Data Modeling**
  - Author:
  - Publication:
  - Date:
  - Link: <https://arxiv.org/pdf/2106.05223>
  - Abstract:
- **A Vertical Federated Learning Framework for Graph Convolutional Network**
  - Author:
  - Publication:
  - Date:
  - Link: <https://arxiv.org/pdf/2106.11593>
  - Abstract:

- **A Federated Multigraph Integration Approach for Connectional Brain Template Learning**
  - Author:
  - Publication:
  - Date:
  - Link: <https://link.springer.com/chapter/10.1007/978-3-030-89847-2_4>
  - Abstract:

***

## FL & Knowledge Graph

- **Differentially Private Federated Knowledge Graphs Embedding**
  - Author: Hao Peng, Haoran Li, Yangqiu Song, Vincent Zheng, Jianxin Li
  - Publication: CIKM 2021
  - Date: 17 May 2021
  - Link: <https://arxiv.org/abs/2105.07615>
  - Abstract:
    Knowledge graph embedding plays an important role in knowledge representation, reasoning, and data mining applications. However, for multiple cross-domain knowledge graphs, state-of-the-art embedding models cannot make full use of the data from different knowledge domains while preserving the privacy of exchanged data. In addition, the centralized embedding model may not scale to the extensive real-world knowledge graphs. Therefore, we propose a novel decentralized scalable learning framework, \emph{Federated Knowledge Graphs Embedding} (FKGE), where embeddings from different knowledge graphs can be learnt in an asynchronous and peer-to-peer manner while being privacy-preserving. FKGE exploits adversarial generation between pairs of knowledge graphs to translate identical entities and relations of different domains into near embedding spaces. In order to protect the privacy of the training data, FKGE further implements a privacy-preserving neural network structure to guarantee no raw data leakage. We conduct extensive experiments to evaluate FKGE on 11 knowledge graphs, demonstrating a significant and consistent improvement in model quality with at most 17.85\% and 7.90\% increases in performance on triple classification and link prediction tasks.
  - Note
    - Knowledge Graph
- **FedE: Embedding Knowledge Graphs in Federated Setting.** [Link](https://arxiv.org/pdf/2010.12882)
- **Improving Federated Relational Data Modeling via Basis Alignment and Weight Penalty.** [Link](https://arxiv.org/pdf/2011.11369)
- **Federated Knowledge Graphs Embedding.**[Link](https://arxiv.org/pdf/2105.07615)
- **Leveraging a Federation of Knowledge Graphs to Improve Faceted Search in Digital Libraries.** [Link](https://arxiv.org/pdf/2107.05447)

<!--
- ****
  - Author: 
  - Publication: 
  - Date: 
  - Link: <>
  - Abstract:
-->
