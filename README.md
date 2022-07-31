# Conditional GAN Experiments Package

Simulate experiments with a Conditional GAN architecture package.

## Setting up the Conditional GAN Experiments package

- Setup the environment - 
```
make setup
```

- Allow the `.envrc` to activate vitual environment - 
```
direnv allow
```

- Install the requirements specified in `requirements.txt` - 
```
make install
```

## Using the Conditional GAN Experiments package

For **every** experiment run:

- Clean the directory structure - 
```
make clean
```

- Reset the directory structure - 
```
make reset
```

- Define `config` parameters for package (each key in `config.py` - `config` dictionary **must** have value)

For **each** experiment run:

- Train the model - 
```
make experiments
```

- Visualize model training - inferred and true data distributions, losses [`Generator`, `Discriminator`, `Discriminator (Real Samples)`, `Discriminator (Fake Samples)`] and generated samples (Uses `Tensorboard`) - 
```
make visualize
```

## Example For Using API

 - Training from scratch - 

```python
def train_from_scratch_example() -> None:
    '''
    Train a model from scratch.
    Returns:
        None
    '''
    # Create directory structure for the experiment
    create_directory_structure = DirectoryStructure(home_dir=config['device']['home directory'])
    create_directory_structure.create_directory_structure()

    # Create the experiments
    experiments = Experiments(config=config)

    # Train the model
    experiments.train(verbose=True, checkpoint=None)
```

- Training from checkpoint - 

```python
def train_from_checkpoint_example() -> None:
    '''
    Train a model from a checkpoint.
    Returns:
        None
    '''
    # Create the experiments
    experiments = Experiments(config=config)

    checkpoint = {
        'generator': './weights/generator_epoch_0_loss_0.pt',
        'discriminator': './weights/discriminator_epoch_0_loss_0.pt'
    }

    # Train the model
    experiments.train(verbose=True, checkpoint=checkpoint)
```

## References:

The *Conditional Generative Adversarial Nets* training algorithm can be found [here](https://arxiv.org/abs/1411.1784v1) - 
```
@misc{https://doi.org/10.48550/arxiv.1411.1784,
  doi = {10.48550/ARXIV.1411.1784},
  url = {https://arxiv.org/abs/1411.1784},
  author = {Mirza, Mehdi and Osindero, Simon},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Conditional Generative Adversarial Nets},
  publisher = {arXiv},
  year = {2014},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
