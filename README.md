# CR-Adaptive and CR-Absolute Models

This repository contains implementations of two models: **CR-Adaptive** and **CR-Absolute**. Both models are designed to operate on graph datasets, with specific configurations for the Chameleon dataset.

## Requirements
Make sure you have all required dependencies installed. You can install them via:

```bash
pip install -r requirements.txt
```

## Running the Models

### CR-Adaptive
To run the **CR-Adaptive** model, execute:

```bash
python train_CR_MAS_exp1.py --dataset chameleon --hops 3 --peak_lr 0.01 --device 1 --save_to folder --num_structure_supernodes 30
```

### CR-Absolute
To run the **CR-Absolute** model, execute:

```bash
python train_CR_TAS_exp1.py --dataset chameleon --hops 3 --peak_lr 0.01 --device 1 --save_to folder --num_structure_supernodes 30
```

### Parameters
- `--dataset`: Name of the graph dataset (e.g., `chameleon`).
- `--hops`: Number of message-passing hops.
- `--peak_lr`: Peak learning rate.
- `--device`: GPU device index.
- `--save_to`: Output directory for saving output results.
- `--num_structure_supernodes`: Number of structural supernodes used in the model.

## File Descriptions
- `train_CR_MAS_exp1.py`: Training script for the CR-Adaptive model.
- `train_CR_TAS_exp1.py`: Training script for the CR-Absolute model.

Please refer to these files for more detailed information on the model configurations and additional input arguments.

## Contact
For any questions or feedback, please contact mle19@jh.edu.
