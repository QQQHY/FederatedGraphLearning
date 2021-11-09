# Federated Graph Learning

Generally sorted by category and date.

***

## FL & GNN

- **FedGNN: Federated Graph Neural Network for Privacy-Preserving Recommendation.**
  - Author: Chuhan Wu, Fangzhao Wu, Yang Cao, Yongfeng Huang, Xing Xie
  - Publication: KDD 2021
  - Date: Tue, 9 Feb 2021
  - Link: <https://arxiv.org/pdf/2102.04925>  
  - Abstract: Graphs have been widely used in data mining and machine learning due to their unique representation of real-world objects and their interactions. As graphs are getting bigger and bigger nowadays, it is common to see their subgraphs separately collected and stored in multiple local systems. Therefore, it is natural to consider the subgraph federated learning setting, where each local system holds a small subgraph that may be biased from the distribution of the whole graph. Hence, the subgraph federated learning aims to collaboratively train a powerful and generalizable graph mining model without directly sharing their graph data. In this work, towards the novel yet realistic setting of subgraph federated learning, we propose two major techniques: (1) FedSage, which trains a GraphSage model based on FedAvg to integrate node features, link structures, and task labels on multiple local subgraphs; (2) FedSage+, which trains a missing neighbor generator along FedSage to deal with missing links across local subgraphs. Empirical results on four real-world graph datasets with synthesized subgraph federated learning settings demonstrate the effectiveness and efficiency of our proposed techniques. At the same time, consistent theoretical implications are made towards their generalization ability on the global graphs.

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
- **Federated Graph Classification over Non-IID Graphs** [Link](https://arxiv.org/abs/2106.13423)
- **Federated Dynamic GNN with Secure Aggregation.** [Link](https://arxiv.org/pdf/2009.07351)
- **Privacy-Preserving Graph Neural Network for Node Classification.** [Link](https://arxiv.org/pdf/2005.11903)
- **ASFGNN: Automated Separated-Federated Graph Neural Network.** [Link](https://arxiv.org/pdf/2011.03248)
- **GraphFL: A Federated Learning Framework for Semi-Supervised Node Classification on Graphs.** [Link](https://arxiv.org/pdf/2012.04187)

- **FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks.** [Link](https://arxiv.org/pdf/2104.07145)
- **FL-AGCNS: Federated Learning Framework for Automatic Graph Convolutional Network Search.** [Link](https://arxiv.org/pdf/2104.04141)
- **Cluster-driven Graph Federated Learning over Multiple Domains.** [Link](https://arxiv.org/pdf/2104.14628)
- **FedGL: Federated Graph Learning Framework with Global Self-Supervision.** [Link](https://arxiv.org/pdf/2105.03170)
- **Federated Graph Learning -- A Position Paper.** [Link](https://arxiv.org/pdf/2105.11099)
- **SpreadGNN: Serverless Multi-task Federated Learning for Graph Neural Networks.** [Link](https://arxiv.org/pdf/2106.02743)
- **Cross-Node Federated Graph Neural Network for Spatio-Temporal Data Modeling.** [Link](https://arxiv.org/pdf/2106.05223)
- **A Vertical Federated Learning Framework for Graph Convolutional Network.** [Link](https://arxiv.org/pdf/2106.11593)
- **Federated Graph Classification over Non-IID Graphs.** [Link](https://arxiv.org/pdf/2106.13423)
- **Subgraph Federated Learning with Missing Neighbor Generation.** [Link](https://arxiv.org/pdf/2106.13430)
- **Differentially Private Federated Knowledge Graphs Embedding.** [Link](https://arxiv.org/pdf/2105.07615)
- **A Federated Multigraph Integration Approach for Connectional Brain Template Learning.** [Link](https://link.springer.com/chapter/10.1007/978-3-030-89847-2_4)
- **Federated Graph Learning -- A Position Paper** [Link](https://arxiv.org/abs/2105.11099)
- FedGL: Federated Graph Learning Framework with Global Self-Supervision [Link](https://arxiv.org/abs/2105.03170)

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
