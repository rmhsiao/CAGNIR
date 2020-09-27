# Click-Attention Graph for Neural Information Retrieval (CAGNIR)

## Overview

This repository contains the implementations of CAGNIR and its related models (i.e. VSM, DSSM, and VPCG).

CAGNIR applies Graph Attention Networks (GAT) on the click Graph and extends it with the multi-view attention mechanism. So queries and documents can aggregate relevant information from their neighbors, and the relationship between nodes can be measured from multiple perspectives, results in a more proper and refined way to utilize click logs than traditional methods for Neural IR and click graph. Through the principle close to pseudo relevance feedback (PRF), CAGNIR can get representation with complete semantics and reduce sparsity in the click log, thereby improving retrieval performance. And finally, given a query, documents are clicked on for it or close to it in the click graph will have a better and more reasonable ranking.

<img src="https://raw.githubusercontent.com/rmhsiao/CAGNIR/demo/CAGNIR.png" style="zoom: 67%;" />

## File Structure

- `expt/`: Scripts for experiments, including model training, model testing and performance measurement.
- `models/`: Implementation of CAGNIR and other related models 
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
