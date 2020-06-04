
//#[macro_use]
//extern crate rustml;

// https://www.youtube.com/watch?v=ueO_Ph0Pyqk
// loss for softmax = -log(predict@correct class) = sum (yjlogy^ j)
// error at output for softmax = <pred> - <actual>
use rustml::Matrix;
use rustml::ops::{MatrixMatrixOps,MatrixScalarOps};
use rustml::ops_inplace::{FunctionsInPlace,MatrixScalarOpsInPlace,MatrixMatrixOpsInPlace};

/// ## LIBRARY STRUCTURE
/// The key struture in Blasphemy is a NeuralNet. It if defined very simply: 
/// ```
/// pub struct NeuralNet {
///     dim_output: usize,
///     layers: Vec<Layer>,
///     pub alpha: f64
/// }
/// ```
struct Linear {
    bias: Matrix::<f64>,
    weights: Matrix::<f64>,
    input: Matrix::<f64>,
    activation: Matrix::<f64>,
    accumulated_err: Matrix::<f64>
}
struct Sigmoid {
    bias: Matrix::<f64>,
    weights: Matrix::<f64>,
    input: Matrix::<f64>,
    activation: Matrix::<f64>,
    accumulated_err: Matrix::<f64>
}
struct Softmax{
}
struct Dropout{
    reject: f64
}

pub enum Layer {
    Linear{bias: Matrix::<f64>, weights:Matrix::<f64>, input: Matrix::<f64>, activation: Matrix::<f64>, accumulated_err: Matrix::<f64>},
    Sigmoid{bias: Matrix::<f64>, weights:Matrix::<f64>, input: Matrix::<f64>, activation: Matrix::<f64>, accumulated_err: Matrix::<f64>},
    Dropout{reject:f64},
    Softmax
}

pub struct NeuralNet {
    dim_output: usize,
    layers: Vec<Layer>,
    pub alpha: f64
}


fn g_prime(activation: &Matrix::<f64>) -> Matrix::<f64>{
    // the derivative of the activation is a*(1-a)
    let inp_deriv = activation.clone(); // d/dx of the activation functions
    inp_deriv.mul_scalar(-1f64).add_scalar(1f64).imule(activation);
    inp_deriv 
}
impl NeuralNet {

    pub fn new(dim_output: usize) -> NeuralNet {
        // Create a new model with input dimensions dim_input
        
        NeuralNet{dim_output:dim_output, layers: Vec::new(), alpha: 0.05f64}
    }

    pub fn linear(&mut self, dim: usize) {
        let mut weights = Matrix::<f64>::random::<f64>(self.dim_output, dim);
        weights.idiv_scalar(100f64);
        let next_layer: Layer = Layer::Linear{
            bias: Matrix::<f64>::random::<f64>(1, dim),
            weights: weights,
            input: Matrix::<f64>::fill(0f64, 1, self.dim_output), // the input from the preceeding layer
            activation: Matrix::<f64>::fill(0f64, 1, dim),
            accumulated_err:  Matrix::<f64>::fill(0f64, self.dim_output, dim)
        };
        self.layers.push(next_layer);
        self.dim_output = dim;
    }

    pub fn sigmoid(&mut self, dim: usize) {
        let mut weights = Matrix::<f64>::random::<f64>(self.dim_output, dim);
        weights.idiv_scalar(100f64);
        let next_layer: Layer = Layer::Sigmoid{
            bias: Matrix::<f64>::random::<f64>(1, dim),
            weights: weights,
            input: Matrix::<f64>::fill(0f64, 1, self.dim_output), // the input from the preceeding layer
            activation: Matrix::<f64>::fill(0f64, 1, dim),
            accumulated_err:  Matrix::<f64>::fill(0f64, self.dim_output, dim)
        };
        self.layers.push(next_layer);
        self.dim_output = dim;
    }

    pub fn softmax(&mut self) {
        let next_layer: Layer = Layer::Softmax{};
        self.layers.push(next_layer);
    }



    pub fn forward_prop(&mut self, input: &Matrix::<f64>) -> Matrix::<f64> {
        let mut x = input.clone();
        for layer in self.layers.iter_mut(){
            match layer {
                Layer::Linear{bias, weights, input, activation, accumulated_err} => {
                    *input = x.clone();
                    x = x.mul(&weights, false, false);
                    x.iadd(&bias);
                    *activation = x.clone();
                },
                Layer::Sigmoid{bias, weights, input,activation, accumulated_err} =>{
                    *input = x.clone();
                    x = x.mul(&weights, false, false);
                    x.iadd(&bias);
                    x.isigmoid();
                    *activation = x.clone();
                },
                Layer::Softmax{} =>{
                    let mut norm: f64 = 0f64;
                    for  xi in x.iter_mut(){
                        *xi = 2.71828f64.powf(*xi);
                        norm = norm + *xi;
                    }
                    x.idiv_scalar(norm);
                },
                Layer::Dropout{reject} => {

                }
            }
        }
    // return x after all the transformations
    x
    }

    pub fn backprop(&mut self, inp: &Matrix::<f64>, output: &Matrix::<f64>) -> f64 {
        // take inp as an input, perform forward prop and backprop, 
        // return the AVERAGE error at tne output layer
        let mut error: Matrix::<f64> = self.forward_prop(inp);
        error.isub(output);
        let mut sum_error = 0f64;
        for ei in error.iter(){
            sum_error = sum_error + (ei*ei);
        }
        for layer in self.layers.iter_mut().rev(){
            match layer {
                Layer::Softmax{} => (),
                Layer::Sigmoid{bias, weights, input,activation, accumulated_err} => {
                    //println!("Shp Error Inbound: {}", &error);
                    //println!("Shp Activation: {}", &activation);
                    let delta_error = input.mul(&error, true, false);
                    //println!("Shp DeltaError: {}", &delta_error);
                    accumulated_err.iadd(&delta_error);
                    // calculate the error for the previous layer
                    error = error.mul(&weights, false, true); // transpose W(l)*error(l+1)
                    let inp_deriv = g_prime(&input);
                    error.imule(&inp_deriv);
                }
                Layer::Linear{bias, weights, input, activation,accumulated_err} => {
                    let delta_error = activation.mul(&error, true, false);
                    accumulated_err.iadd(&delta_error);
                    // calculate the error for the previous layer
                    error = error.mul(&weights, false, true); // transpose W(l)*error(l+1)
                    let inp_deriv = g_prime(&input);
                    error.imule(&inp_deriv);
                }
                Layer::Dropout{reject} => ()
            }
            
        }
        sum_error 
    }

    pub fn grad_descent(&mut self){
        for layer in self.layers.iter_mut().rev(){
            match layer {
                Layer::Softmax{} => (),
                Layer::Sigmoid{bias, weights, input,activation, accumulated_err} => {
                    accumulated_err.mul_scalar(-0.05f64);
                    weights.iadd(accumulated_err);
                    weights.mul_scalar(0f64);
                }
                Layer::Linear{bias, weights, input, activation,accumulated_err} => {
                    accumulated_err.mul_scalar(-0.05f64);
                    weights.iadd(accumulated_err);
                    weights.mul_scalar(0f64);
                }
                Layer::Dropout{reject} => ()
            }
            
        }

    }
}


fn main() {

    let mut nn = NeuralNet::new(10); 
    //nn.linear(20);
    nn.sigmoid(12);
    nn.sigmoid(5);
    nn.sigmoid(4);
    nn.softmax();

    for iter in 0..1000{
        let input = Matrix::<f64>::random::<f64>(1,10);
        let output = Matrix::<f64>::random::<f64>(1,4);
        let err = nn.backprop(&input, &output);
        nn.grad_descent();
        println!("iter={} error={}", iter, err);
    }
    //println!("OUTPUT = {}", out);
    //println!("INOUT={}", input);
    

}

//let b = Matrix::<T>::fill(2f64, 300, 1_000);
//let a = mat![1.0, 2.0; 3.0, 4.0];