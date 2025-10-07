# CR-Adaptive and CR-Absolute Models

This repository contains implementations of **CR-Adaptive** (CR-MAS) and **CR-Absolute** (CR-TAS), as well as other baselines on MAS and TAS auxiliary graphs. Both models are designed to operate on graph datasets.

## Requirements
Make sure you have all the required dependencies installed. You can install them via:

```bash
pip install -r requirements.txt
```

## Generating Auxiliary Graphs

### MAS Auxiliary Graph
To generate the MAS auxiliary graph, for example, execute:

```bash
python generator_MAS_exp1.py --dataset chameleon
```

### TAS Auxiliary Graph
To generate the TAS auxiliary graph, for example, execute:

```bash
python generator_TAS_exp1.py --dataset chameleon
```

## Running the Models

### CR-Adaptive (CR-MAS)
To run the **CR-Adaptive** model, for example, execute:

```bash
python train_CR_MAS_exp1.py --dataset chameleon --peak_lr 0.01 --device 1 --save_to folder
```

### CR-Absolute (CR-TAS)
To run the **CR-Absolute** model, for example, execute:

```bash
python train_CR_TAS_exp1.py --dataset chameleon --peak_lr 0.01 --device 1 --save_to folder
```

### Parameters
- `--dataset`: Name of the graph dataset (e.g., `chameleon`).
- `--peak_lr`: Peak learning rate.
- `--device`: GPU device index.
- `--save_to`: Output directory for saving results.

## File Descriptions
- `train_CR_MAS_exp1.py`: Training script for the CR-Adaptive model.
- `train_CR_TAS_exp1.py`: Training script for the CR-Absolute model.
- `train_GCN_MAS_exp1.py`: Training script for the GCN model on the MAS auxiliary graph.
- `train_GCN_TAS_exp1.py`: Training script for the GCN model on the TAS auxiliary graph.
- `train_GAT_MAS_exp1.py`: Training script for the GAT model on the MAS auxiliary graph.
- `train_GAT_TAS_exp1.py`: Training script for the GAT model on the TAS auxiliary graph.
- `train_GraphSAGE_MAS_exp1.py`: Training script for the GraphSAGE model on the MAS auxiliary graph.
- `train_GraphSAGE_TAS_exp1.py`: Training script for the GraphSAGE model on the TAS auxiliary graph.
- `train_GraphGPS_MAS_exp1.py`: Training script for the GraphGPS model on the MAS auxiliary graph.
- `train_GraphGPS_TAS_exp1.py`: Training script for the GraphGPS model on the TAS auxiliary graph.
- `train_SAN_MAS_exp1.py`: Training script for the SAN model on the MAS auxiliary graph.
- `train_SAN_TAS_exp1.py`: Training script for the SAN model on the TAS auxiliary graph.
- `train_VCR_MAS_exp1.py`: Training script for the VCR model on the MAS auxiliary graph.
- `train_VCR_TAS_exp1.py`: Training script for the VCR model on the TAS auxiliary graph.
- `train_NAG_MAS_exp1.py`: Training script for the NAG model on the MAS auxiliary graph.
- `train_NAG_TAS_exp1.py`: Training script for the NAG model on the TAS auxiliary graph.

Other models use syntax similar to the CR-Adaptive and CR-Absolute models. Please refer to these files for more detailed information on model configurations and additional input arguments.
