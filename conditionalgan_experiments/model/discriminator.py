import torch
import torch.nn as nn

import typing
import numpy as np

import warnings

class Discriminator(nn.Module):
    def __init__(self, in_shape: tuple, n_blocks: int, num_classes: int, name:str=None) -> None:
        '''
        Initialize the discriminator.
        Parameters:
            in_shape: The shape of the input image.
            n_blocks: The number of blocks in the discriminator.
            num_classes: The number of classes in the dataset.
            name: The name of the discriminator.
        Returns:
            None
        '''
        super(Discriminator, self).__init__()
        self.name = "Discriminator" if name is None else name
        self.in_shape = in_shape
        self.n_blocks = n_blocks
        self.num_classes = num_classes
        
        def block(in_features, out_features, normalize=True, regularize=True) -> typing.List[nn.Module]:
            '''
            Each block that makes up the discriminator.
            Parameters:
                in_features: The input features of the block.
                out_features: The output features of the block.
                normalize: Whether or not to add batch normalization.
                regularize: Whether or not to add regularization.
            Returns:
                A list of modules that make up the block.
            '''
            # Fully connected layer
            layers = [nn.Linear(in_features=in_features, out_features=out_features)]

            if normalize:
                # Batch normalization layer
                layers.append(nn.BatchNorm1d(num_features=out_features, eps=0.8))
            
            # Activation layer
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            if regularize:
                # Regularization layer
                layers.append(nn.Dropout(p=0.5))
            
            return layers
        
        # Starting intermediate latent dimension
        self.inter_dim = 512

        # Define pre block
        self.pre_in_block = nn.ModuleDict({
            'embedding_layer': nn.Sequential(nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.num_classes))
        })

        # Define input block
        self.in_block = nn.ModuleDict({
            'in_block': nn.Sequential(*block(in_features=int(np.prod(in_shape)), out_features=self.inter_dim, normalize=False, regularize=False))
        })

        # Define intermediate blocks
        self.inter_blocks = nn.ModuleDict({})
        for i in range(self.n_blocks):
            self.inter_blocks[f'inter_block_{i+1}'] = nn.Sequential(*block(in_features=self.inter_dim, out_features=self.inter_dim, normalize=True, regularize=True))
            
        # Define output block
        self.out_block = nn.ModuleDict({
            'out_block': nn.Sequential(
                nn.Linear(in_features=self.inter_dim, out_features=1),
                nn.Sigmoid())
        })

        # Initialize weights
        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, m: nn.Module) -> None:
        '''
        Initialize the weights of the discriminator.
        Parameters:
            m: The module to initialize.
        Returns:
            None
        '''
        if isinstance(m, nn.Linear):
            # Initialize weight to random normal
            nn.init.xavier_normal_(m.weight)
            # Initialize bias to zero
            nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            # Initialize weight to random normal
            nn.init.xavier_normal_(m.weight)
        
    def forward(self, imgs: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        '''
        Forward pass of the discriminator.
        Parameters:
            x: The input image.
            labels: The labels of the input image.
        Returns:
            The output score.
        '''
        # Reshape input
        imgs = imgs.view(imgs.size(0), -1)
        x = torch.cat((imgs, self.pre_in_block['embedding_layer'](labels)), -1)

        # Input block
        x = self.in_block['in_block'](x)

        # Intermediate blocks
        for i in range(self.n_blocks):
            x = self.inter_blocks[f'inter_block_{i+1}'](x)
        
        # Output block
        validity = self.out_block['out_block'](x)
        
        return validity