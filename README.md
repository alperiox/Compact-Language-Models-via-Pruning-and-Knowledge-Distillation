# Compact Language Models via Pruning and Knowledge Distillation

This project is an unofficial implementation of the paper [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/pdf/2407.14679). It explores techniques for compressing large language models (LLMs) through a combination of pruning and knowledge distillation.

## Overview

The goal of this project is to investigate whether pruning an existing LLM and then re-training it with a small fraction of the original training data can be a viable alternative to training each model variant from scratch. The implementation focuses on:

1. Pruning strategies for width, attention, and MLP layers
2. Combining different pruning axes
3. Knowledge distillation techniques for retraining
4. Searching for optimal compressed architectures

## Project Structure

- `models.py`: Contains the implementation of the GPT model and its components
- `hooks.py`: Implements forward hooks for calculating importance scores
- `pruners.py`: Contains functions for pruning neurons, attention heads, and embeddings
- `utils.py`: Utility functions for data loading, model saving/loading, and evaluation
- `script.py`: Main script for running experiments

## Getting Started

1. Clone the repository
2. Install the required dependencies (list them here or include a `requirements.txt` file)
3. Download the training data (Shakespeare dataset) by running the script
4. Adjust hyperparameters in `script.py` as needed
5. Run `script.py` to train the base model and perform pruning experiments

## Key Features

- Implementation of a GPT-style language model
- Flexible pruning strategies for different model components
- Knowledge distillation for model retraining
- Experimental framework for testing various compression configurations

## Usage

The implementation doesn't support any kind of CLI usage, I kind of got focused on the math heavy stuff.

## Results

(work in progress)

## Limitations and Future Work

- This implementation currently focuses on a smaller scale model compared to the paper (like, a few thousand times smaller since I don't got any GPUs?)
- Further optimization of pruning and distillation techniques may be possible (didn't implement depth pruning as my focus is applying the technique on smaller models <15B)

## Acknowledgements

```
@article{minitron2024,
      title={Compact Language Models via Pruning and Knowledge Distillation}, 
      author={Saurav Muralidharan and Sharath Turuvekere Sreenivas and Raviraj Joshi and Marcin Chochowski and Mostofa Patwary and Mohammad Shoeybi and Bryan Catanzaro and Jan Kautz and Pavlo Molchanov},
      journal={arXiv preprint arXiv:2407.14679},
      year={2024},
      url={https://arxiv.org/abs/2407.14679}, 
}
```

- [Andrej Karpathy](https://github.com/karpathy) for literally firing me up for working on my FOMO.

## References

[Original paper](https://arxiv.org/pdf/2407.14679)