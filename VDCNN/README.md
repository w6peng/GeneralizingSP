# Deep Reinforcement Learning for Reliable NLP Networks

Implementation of "Deep Reinforcement Learning for Reliable Networks" for use with the VDCNN network architecture. Credit to Kunping Huang for his initial DRL architecture
- Deep Reinforcement Learning for Reliable Networks: [Paper](https://arxiv.org/abs/2001.03814), [Code](http://132.239.170.202/STARGROUP/deep-reinforcement-learning-for-reliable-networks)
- Very Deep Convolutional Neural Network: [Paper](https://arxiv.org/abs/1606.01781), [Code (unofficial)](https://github.com/ArdalanM/nlp-benchmarks)

## Installation

To install, clone git repository or download compressed file & unpack in your desired development environment.

```bash
git clone http://132.239.170.202/STARGROUP/deep-reinforcement-learning-for-reliable-nlp-networks.git
```

## Usage

### DRL
To implement the **TopBits** Selective Protection Technique, run the *rl_length_noise.py* file from the root directory.

To implement the **BitMask** Selective Protection Technique, run the *rl_noise.py* file from the root directory.

#### Runtime Arguments
A number of runtime arguments are available for the DRL environment, the DRL Network, the noise addition, network training, and the model architecture. The full list of arguments can be seen in either *rl_noise.py* or *rl_length_noise.py*, but notable arguments include:

**&dash;&dash;preserve_ratio (default: 0.25)**
- The ratio of bits to protect in relation to the total number of bits in the network parameters

**&dash;&dash;min_bit (default: 0)**
- The minimum left-most bit to consider for protection

**&dash;&dash;max_bit (default: 32)**
- The maximum left-most bit to consider for protection. Note that since this defaults to 32, this parameter must be changed to 8 when using the fixed-point representation

**&dash;&dash;train_episode (default: 5000)**
- Number of training cycles for DRL 

**&dash;&dash;representation (default: 'float')**
- The representation type of the weights. Choices include 'float' or 'fixed'

**&dash;&dash;code (default: 'ideal')**
- The type of ECC to apply to weights for protection. Choices include 'ideal' or 'bch'

**&dash;&dash;history (default: True)**
- Specifies if the change in best_policy information should be displayed after running

### Noise Addition
For testing network performance under noise simulation, the *noise_prob.py* and *noise_prob_layer.py* files can be used. 
- *noise_prob.py* has default layer-wise masks that were obtained through the DRL training for length of 5,000 cycles at a preserve_ratio of 0.25. It can be used to apply layer-wise masks or to subject the entire network to unprotected noise addition
- *noise_prob_layer.py* does not have special masks set. It is to be used for layerwise protection.

Runtime arguments include:

| Argument                                  | noise_prob.py | noise_prob_layerwise.py | Description                                                                    |
|-------------------------------------------|---------------|-------------------------|--------------------------------------------------------------------------------|
| &dash;&dash;cycles (default: 10)          | &#10003;      | &#10003;                | Specifies number of cycles to average noise over                               |
| &dash;&dash;protection                    | &#10003;      | &#10003;                | Type of protection technique ('topbits', 'bitmask', 'baseline')                |
| &dash;&dash;preserve_ratio (default: 0.25)| &#10003;      | &#10007;                | Preserve ratio to calculate baseline performance for                           |
| &dash;&dash;code                          | &#10003;      | &#10007;                | Type of ECC to use ('bch', 'ideal')                                            |
| &dash;&dash;representation                | &#10003;      | &#10003;                | Binary representation type ('fixed', 'float')                                  |
| &dash;&dash;bit                           | &#10003;      | &#10003;                | # bits in representation                                                       |
| &dash;&dash;n_layers (default: 15)        | &#10003;      | &#10003;                | # layers subjected to noise (15 for default VDCNN)                             |
| &dash;&dash;save                          | &#10003;      | &#10003;                | Pass file to save accuracy statistics to                                       |
| &dash;&dash;ber (default: 'standard')     | &#10003;      | &#10007;                | Range at which to add noise ('range': ranges 1e-10 to 5e-4', 'standard': 1e-2) |
| &dash;&dash;dataset                       | &#10003;      | &#10003;                | Dataset to add noise to ('ag_news', 'sogou_news', 'yelp_polarity')             |
| &dash;&dash;dict.                         | &#10003;      | &#10007;                | Dataset to use mask from ('ag_news', 'sogou_news', 'yelp_polarity')            |

### Graphing
A handful of Python notebooks are present in /results/graphers/... These notenooks are organized by graph type and protection type. *grapher.py* additionally contains most of the graphing functions used. These graphing functions are a collection of matplotlib functions that are packaged in functions for easier use and code readability.