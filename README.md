# Click-Attention Graph for Neural Information Retrieval (CAGNIR)

## Overview

CAGNIR applies Graph Attention Networks (GAT) on the Click Graph and extends it with the multi-view attention mechanism. So queries and documents can aggregate relevant information from their neighbors, and the relationship between nodes can be measured from multiple perspectives, results in a more proper and refined way than traditional methods. Through the principle close to pseudo relevance feedback (PRF), CAGNIR can get representation with complete semantics and reduce sparsity in the click log, thereby improving retrieval performance. And finally, given a query, documents are clicked on for it or close to it in the Click Graph will have a better and more reasonable ranking.

**Model Architecture**

| CAGNIR | Click-Attention Graph |
| ---- | ---- |
| ![](https://raw.githubusercontent.com/rmhsiao/CAGNIR/demo/CAGNIR.png) | ![](https://raw.githubusercontent.com/rmhsiao/CAGNIR/demo/Click-Attention%20Graph.png) |

**Retrieval Performance**

| Models                                                     | NDCG@1   | NDCG@3   | NDCG@5   | NDCG@10  |
| ---------------------------------------------------------- | -------- | -------- | -------- | -------- |
| [VSM](https://dl.acm.org/doi/10.1145/361219.361220) <sup>a</sup>    | 0.5081 - | 0.4674 - | 0.4399 - | 0.3941 - |
| VSM <sup>b</sup>                                                    | 0.5484 - | 0.5260 - | 0.5074 - | 0.4755 - |
| [DSSM](https://dl.acm.org/doi/10.1145/2505515.2505665) <sup>a</sup> | 0.5437 - | 0.5033 - | 0.4740 - | 0.4254 - |
| [VPCG](https://dl.acm.org/doi/10.1145/2911451.2911531) <sup>b</sup> | 0.7195 - | 0.7168 - | 0.7208 - | 0.7337 - |
| CAGNIR <sup>a</sup>                                                 | 0.7684   | 0.7622   | 0.7551   | 0.7463   |

*Note.* The dataset [Sogou-QCL](https://dl.acm.org/doi/10.1145/3209978.3210092) is used for experiments, and the results marked with "-" are significantly weaker than CAGNIR at the same position under the Student's Paired t-test (p-value<0.01).

<sup>a,b</sup> The vocabularies used for the model are derived respectively from data through [SentencePiece](https://github.com/google/sentencepiece/) (BPE) and [Jieba](https://github.com/fxsjy/jieba).


## File Structure

- `expt/`: Scripts for experiments, including model training, model testing and performance measurement.
- `models/`: Implementation of CAGNIR and other related models (i.e. VSM, DSSM, VPCG).
- `utils/`:
    - `common/`: Common utilities like interface of database, logger, etc.
    - `data/`: Utilities for data processing.
    - `models/`: Components of models.

## Dependencies

-   `Python 3.6.8`
-   `TensorFlow 1.13.0`
-   `Scikit-learn 0.21.2`
-   `Scipy 1.3.0`
-   `Numpy 1.16.4`
