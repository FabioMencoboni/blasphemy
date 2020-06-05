# blasphemy

The safety of Rust. The raw speed of BLAS. An ergonomic interface for neural network architecture inspired by Keras. That's **blas**phemy.


**Blasphemy is early Work In Progress** - a functionality roadmap is below. It is heavily based on RustML with an emphasis on ergonimics and numeric stability with vanishing gradients.

#### Feature Roadmap
- [x] Random initialization for symmetry breaking
- [x] Linear Layers
- [x] Softmax Layers
- [_] Sigmoid Layers
- [_] ReLU Layers
- [_]  Residual Layers

### Quickstart

You will need BLAS to use blasphemy. If you are using Ubuntu/Debian this can be accomplished with

```
$ sudo apt-get install libblas-dev libopencv-highgui-dev libopenblas-dev
```

#### Hello World example

This example creates a neural network with three linear layers and softmax normalization.
 
```
use blasphemy::{NeuralNet, Matrix};

let mut nn = NeuralNet::new(4); // new Neural Net with input shape (1,4)
nn.linear(5); // add a linear activation layer with 5 neurons
nn.linear(5); // add a linear activation layer with 5 neurons
nn.linear(3); // add a linear activation layer with 3 neurons
nn.softmax(); // apply softmax normalization to the output

let mut err = 0f64;
for iter in 0..200{
    // train on two examples for 200 epochs
    let input = Matrix::from_vec(vec![1f64,0f64,0f64,1f64], 1,4);
    let output = Matrix::from_vec(vec![1f64,0f64,0f64], 1,3);
    err = nn.backprop(&input, &output); // accumulate errors for one example
    let input = Matrix::from_vec(vec![0f64,2f64,2f64,0f64], 1,4);
    let output = Matrix::from_vec(vec![0f64,1f64,0f64], 1,3);
    err = nn.backprop(&input, &output); // accumulate errors for one example
    if iter % 3 == 0 {
    	// every 3rd example, perform gradient descent with the accumulated errors
        nn.grad_descent();
        println!("iter={} error={}", iter, err);
    }
}
// make a prediction
let input = Matrix::from_vec(vec![1f64,0f64,0f64,1f64], 1,4);
let prediction = nn.forward_prop(&input);
```

Note numeric stability can be troublesome with vanishing gradients past ~3-4 layers: this is an item for future improvement, especially with the incorporation of residual layers.