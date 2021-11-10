# Graph Learning

Generally sorted by date.

***

- **Modeling Relational Data with Graph Convolutional Networks**
  - Author: Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling
  - Publication: ESWC 2018
  - Date: 17 Mar 2017
  - Link: <https://arxiv.org/abs/1703.06103>
  - Abstract: Knowledge graphs enable a wide variety of applications, including question answering and information retrieval. Despite the great effort invested in their creation and maintenance, even the largest (e.g., Yago, DBPedia or Wikidata) remain incomplete. We introduce Relational Graph Convolutional Networks (R-GCNs) and apply them to two standard knowledge base completion tasks: Link prediction (recovery of missing facts, i.e. subject-predicate-object triples) and entity classification (recovery of missing entity attributes). R-GCNs are related to a recent class of neural networks operating on graphs, and are developed specifically to deal with the highly multi-relational data characteristic of realistic knowledge bases. We demonstrate the effectiveness of R-GCNs as a stand-alone model for entity classification. We further show that factorization models for link prediction such as DistMult can be significantly improved by enriching them with an encoder model to accumulate evidence over multiple inference steps in the relational graph, demonstrating a large improvement of 29.8% on FB15k-237 over a decoder-only baseline.
  - Note
    - Heterogeneous graph neural network
    - R-GCN

- **Heterogeneous Graph Neural Network**
  - Author: Zhang, Chuxu and Song, Dongjin and Huang, Chao and Swami, Ananthram and Chawla, Nitesh V.
  - Publication: SIGKDD 2019
  - Date: 25 July 2019
  - Link: <https://dl.acm.org/doi/abs/10.1145/3292500.3330961>
  - Abstract: Representation learning in heterogeneous graphs aims to pursue a meaningful vector representation for each node so as to facilitate downstream applications such as link prediction, personalized recommendation, node classification, etc. This task, however, is challenging not only because of the demand to incorporate heterogeneous structural (graph) information consisting of multiple types of nodes and edges, but also due to the need for considering heterogeneous attributes or contents (e.g., text or image) associated with each node. Despite a substantial amount of effort has been made to homogeneous (or heterogeneous) graph embedding, attributed graph embedding as well as graph neural networks, few of them can jointly consider heterogeneous structural (graph) information as well as heterogeneous contents information of each node effectively. In this paper, we propose HetGNN, a heterogeneous graph neural network model, to resolve this issue. Specifically, we first introduce a random walk with restart strategy to sample a fixed size of strongly correlated heterogeneous neighbors for each node and group them based upon node types. Next, we design a neural network architecture with two modules to aggregate feature information of those sampled neighboring nodes. The first module encodes "deep" feature interactions of heterogeneous contents and generates content embedding for each node. The second module aggregates content (attribute) embeddings of different neighboring groups (types) and further combines them by considering the impacts of different groups to obtain the ultimate node embedding. Finally, we leverage a graph context loss and a mini-batch gradient descent procedure to train the model in an end-to-end manner. Extensive experiments on several datasets demonstrate that HetGNN can outperform state-of-the-art baselines in various graph mining tasks, i.e., link prediction, recommendation, node classification & clustering and inductive node classification & clustering.

- **Subgraph Neural Networks**
  - Author: Emily Alsentzer, Samuel G. Finlayson, Michelle M. Li, Marinka Zitnik
  - Publication: NeurIPS 2020
  - Date: 18 Jun 2020
  - Link: <https://arxiv.org/abs/2006.10538>
  - Abstract: Deep learning methods for graphs achieve remarkable performance on many node-level and graph-level prediction tasks. However, despite the proliferation of the methods and their success, prevailing Graph Neural Networks (GNNs) neglect subgraphs, rendering subgraph prediction tasks challenging to tackle in many impactful applications. Further, subgraph prediction tasks present several unique challenges: subgraphs can have non-trivial internal topology, but also carry a notion of position and external connectivity information relative to the underlying graph in which they exist. Here, we introduce SubGNN, a subgraph neural network to learn disentangled subgraph representations. We propose a novel subgraph routing mechanism that propagates neural messages between the subgraph's components and randomly sampled anchor patches from the underlying graph, yielding highly accurate subgraph representations. SubGNN specifies three channels, each designed to capture a distinct aspect of subgraph topology, and we provide empirical evidence that the channels encode their intended properties. We design a series of new synthetic and real-world subgraph datasets. Empirical results for subgraph classification on eight datasets show that SubGNN achieves considerable performance gains, outperforming strong baseline methods, including node-level and graph-level GNNs, by 19.8% over the strongest baseline. SubGNN performs exceptionally well on challenging biomedical datasets where subgraphs have complex topology and even comprise multiple disconnected components.
  - Note
    - [Subgraph Neural Networks论文阅读](https://zhuanlan.zhihu.com/p/337203750)
  
<!--
- ****
  - Author: 
  - Publication: 
  - Date: 
  - Link: <>
  - Abstract:
-->
