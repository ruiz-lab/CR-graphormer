# Graph Cascades

This repository provides implementations of Graph Cascades based on Maximum Adjacency Search (MAS) and Threshold Adjacency Search (TAS) dynamics, together with an end-to-end node classification framework built on Graph Cascades and a range of baseline models, including GCN, GraphSAGE, GAT, GT, Gophormer, Graphormer, SAN, GraphGPS, NAGphormer, Exphormer, and VCR-Graphormer. All models are designed to operate on graph-structured datasets.


## Requirements
Make sure you have all the required dependencies installed. You can install them via:

```bash
pip install -r requirements.txt
```


## Generating Auxiliary Graphs

### MAS Auxiliary Graph
To generate the MAS auxiliary graph, for example, execute:

```bash
python generator_MAS_exp1.py \
    --dataset chameleon \
    --seed 0 \
    --p 1.0 \
    --l 10 \
    --num_start_nodes 5 \
    --num_permutations 5
```

### TAS Auxiliary Graph
To generate the TAS auxiliary graph, for example, execute:

```bash
python generator_TAS_exp1.py \
    --dataset chameleon \
    --seed 0 \
    --p 1.0 \
    --l 10 \
    --num_start_nodes 5 \
    --num_permutations 5 \
    --threshold_list 1,2,3,4,5
```

### Parameter Description
- `--dataset` *(str)*: Name of the dataset (e.g., `cora`, `chameleon`).
- `--seed` *(int)*: Random seed for reproducibility.
- `--p` *(float)*: Lazy activation parameter controlling cascade propagation.
- `--l` *(int)*: Maximum length of walks.
- `--num_start_nodes` *(int)*: Number of starting neighbors for each cascade.
- `--num_permutations` *(int)*: Number of random permutations used in cascade generation.


## Running the Models

### CR-MAS
To run the **CR-MAS** model, for example, execute:

```bash
python train_CR_MAS_exp1.py \
    --dataset chameleon \
    --device 0 \
    --save_to folder \
    --k 5
```

### CR-TAS
To run the **CR-TAS** model, for example, execute:

```bash
python train_CR_TAS_exp1.py \
    --dataset chameleon \
    --device 0 \
    --save_to folder \
    --k 5
```

### Main Parameters

#### General
- `--dataset` *(str)*: Name of the dataset (e.g., `cora`, `chameleon`).
- `--save_to` *(str)*: Directory for saving results.
- `--device` *(int)*: CUDA device ID.
- `--seed` *(int)*: Random seed for reproducibility.
- `--adj_renorm` *(bool)*: Whether to normalize the input adjacency matrix.
- `--merge` *(bool)*: Whether to merge the auxiliary graph with the original graph.
- `--p` *(float)*: Lazy activation parameter.
- `--k` *(int)*: Top-`k` parameter.
- `--normalization` *(str)*: Frequency normalization method.

#### Model Parameters
- `--num_structure_tokens` *(int)*: Number of structure-aware virtually connected neighbors.
- `--num_content_tokens` *(int)*: Number of content-aware virtually connected neighbors.
- `--hidden_dim` *(int)*: Hidden layer dimension.
- `--n_layers` *(int)*: Number of transformer layers.
- `--n_heads` *(int)*: Number of attention heads.
- `--dropout` *(float)*: Dropout rate.
- `--attention_dropout` *(float)*: Dropout rate in attention layers.

#### Training Parameters
- `--batch_size` *(int)*: Batch size.
- `--epochs` *(int)*: Number of training epochs.
- `--tot_updates` *(int)*: Total number of updates for learning rate scheduling.
- `--warmup_updates` *(int)*: Number of warmup steps.
- `--peak_lr` *(float)*: Peak learning rate.
- `--end_lr` *(float)*: Final learning rate.
- `--weight_decay` *(float)*: Weight decay.
- `--patience` *(int)*: Patience for early stopping.
- `--train_size` *(float)*: Proportion of training data.
- `--val_size` *(float)*: Proportion of validation data.

## File Descriptions

### Cascade Models
We refer to **cascade models** as models that take the auxiliary graph \(G'\), along with cascade frequency–based edge weights \(W^\star\), as inputs. Our cascade models include **GCN-Cascade, GraphGPS-Cascade, NAG-Cascade,** and **CR-Cascade**.

- `train_GCN_MAS_exp1.py`: Training script for the GCN-MAS cascade model.
- `train_GCN_TAS_exp1.py`: Training script for the GCN-TAS cascade model.
- `train_GraphGPS_MAS_exp1.py`: Training script for the GraphGPS-MAS cascade model.
- `train_GraphGPS_TAS_exp1.py`: Training script for the GraphGPS-TAS cascade model.
- `train_NAG_MAS_exp1.py`: Training script for the NAG-MAS cascade model.
- `train_NAG_TAS_exp1.py`: Training script for the NAG-TAS cascade model.
- `train_VCR_MAS_exp1.py`: Training script for the VCR-MAS cascade model.
- `train_VCR_TAS_exp1.py`: Training script for the VCR-TAS cascade model.
- `train_CR_MAS_exp1.py`: Training script for the CR-MAS cascade model.
- `train_CR_TAS_exp1.py`: Training script for the CR-TAS cascade model.

### Baseline Models with Auxiliary Graph Input
These models take only the auxiliary graph \(G'\) as input, following the standard architecture of the corresponding baseline (e.g., GraphSAGE, GAT, etc.).  

- `train_GraphSAGE_MAS_exp1.py`: Training script for GraphSAGE with MAS auxiliary graph.
- `train_GraphSAGE_TAS_exp1.py`: Training script for GraphSAGE with TAS auxiliary graph.
- `train_GAT_MAS_exp1.py`: Training script for GAT with MAS auxiliary graph.
- `train_GAT_TAS_exp1.py`: Training script for GAT with TAS auxiliary graph.
- `train_GT_MAS_exp1.py`: Training script for GT with MAS auxiliary graph.
- `train_GT_TAS_exp1.py`: Training script for GT with TAS auxiliary graph.
- `train_Gophormer_MAS_exp1.py`: Training script for Gophormer with MAS auxiliary graph.
- `train_Gophormer_TAS_exp1.py`: Training script for Gophormer with TAS auxiliary graph.
- `train_Graphormer_MAS_exp1.py`: Training script for Graphormer with MAS auxiliary graph.
- `train_Graphormer_TAS_exp1.py`: Training script for Graphormer with TAS auxiliary graph.
- `train_SAN_MAS_exp1.py`: Training script for SAN with MAS auxiliary graph.
- `train_SAN_TAS_exp1.py`: Training script for SAN with TAS auxiliary graph.
- `train_Exphormer_MAS_exp1.py`: Training script for Exphormer with MAS auxiliary graph.
- `train_Exphormer_TAS_exp1.py`: Training script for Exphormer with TAS auxiliary graph.

### Standard Baseline Models
These models operate on the **original input graph** without using auxiliary graphs.

- `train_GCN_exp1.py`: Training script for the GCN model.
- `train_GraphSAGE_exp1.py`: Training script for the GraphSAGE model.
- `train_GAT_exp1.py`: Training script for the GAT model.
- `train_GT_exp1.py`: Training script for the GT model.
- `train_Gophormer_exp1.py`: Training script for the Gophormer model.
- `train_Graphormer_exp1.py`: Training script for the Graphormer model.
- `train_SAN_exp1.py`: Training script for the SAN model.
- `train_GraphGPS_exp1.py`: Training script for the GraphGPS model.
- `train_NAG_exp1.py`: Training script for the NAGphormer model.
- `train_Exphormer_exp1.py`: Training script for the Exphormer model.
- `train_VCR_exp1.py`: Training script for the VCR-Graphormer model.

All other models use a syntax analogous to **CR-MAS** and **CR-TAS**. For detailed information on model configurations and additional command-line arguments, please consult the respective training scripts.
