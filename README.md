# Click-Attention Graph for Neural Information Retrieval (CAGNIR)

## Overview

CAGNIR applies Graph Attention Networks (GAT) on the Click Graph and extends it with the multi-view attention mechanism. So queries and documents can aggregate relevant information from their neighbors, and the relationship between nodes can be measured from multiple perspectives, results in a more proper and refined way than traditional methods. Through the principle close to pseudo relevance feedback (PRF), CAGNIR can get representation with complete semantics and reduce sparsity in the click log, thereby improving retrieval performance. And finally, given a query, documents are clicked on for it or close to it in the Click Graph will have a better and more reasonable ranking.

### Model Architecture

| CAGNIR | Click-Attention Graph |
| ---- | ---- |
| ![](https://raw.githubusercontent.com/rmhsiao/CAGNIR/demo/CAGNIR.png) | ![](https://raw.githubusercontent.com/rmhsiao/CAGNIR/demo/Click-Attention%20Graph.png) |

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