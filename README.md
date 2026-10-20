# Neural Network from Scratch: MNIST Digit Classifier

A handwritten-digit classifier built with only NumPy. The forward pass, backpropagation and SGD training loop are written by hand, with no ML framework.

## Results

- **Architecture:** 784 → 128 → 64 → 10
- **Test accuracy:** **[96.01]%** on the 10,000 MNIST test images
- **Training:** [1] epoch of per-sample SGD on 60,000 images, learning rate 0.01

## How it works

- **Layers:** fully connected, ReLU on hidden layers, raw logits at the output, softmax for probabilities
- **Initialisation:** He initialisation (weights scaled by √(2 / inputs)), zero biases
- **Loss:** softmax with cross-entropy, so the output delta is `softmax(logits) − one_hot(label)`
- **Backprop:** each layer's weight gradient is the outer product of its delta and its input; the delta passes down through `Wᵀ` and the ReLU derivative
- **Training:** SGD with batch size 1, pixels scaled to 0–1

## Try your own digit

`grayscale_converter.py` turns a photo or drawing into the 784-value MNIST format: grayscale, invert if needed, crop to the digit, resize to fit 20×20, and centre it on a 28×28 canvas. Use `preview=True` to check the result before classifying it.

## Setup

```
pip install numpy pandas pillow matplotlib
python main.py
```

Download `mnist_train.csv` and `mnist_test.csv` from [(https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)] into the project folder (too large to include here).

## What I learned

- Most backprop bugs were shape bugs: a weight gradient is built from a layer's delta and that layer's *input*.
- A network can run without errors and still not learn: ReLU on the output layer, off-by-one labels and unscaled pixels all did this to me.
- Per-sample updates are too small for a GPU to help. My first MLX version was slower than NumPy on the CPU because launch overhead outweighed the maths.

## Next steps

- Mini-batch training, multiple epochs with shuffling, saving weights.
- Implement reinforcement learning and train this model to play simple games.
- Small character-level transformer built from scratch.
