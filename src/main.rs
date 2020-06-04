
//#[macro_use]
//extern crate rustml;

// https://www.youtube.com/watch?v=ueO_Ph0Pyqk
// loss for softmax = -log(predict@correct class) = sum (yjlogy^ j)
// error at output for softmax = <pred> - <actual>
use rustml::Matrix;
use rustml::ops::MatrixMatrixOps;
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
    accumulated_err: Matrix::<f64>
}
struct Sigmoid {
    bias: Matrix::<f64>,
    weights: Matrix::<f64>,
    accumulated_err: Matrix::<f64>
}
struct Softmax{
}
struct Dropout{
    reject: f64
}

pub enum Layer {
    Linear{bias: Matrix::<f64>, weights:Matrix::<f64>, accumulated_err: Matrix::<f64>},
    Sigmoid{bias: Matrix::<f64>, weights:Matrix::<f64>, accumulated_err: Matrix::<f64>},
    Dropout{reject:f64},
    Softmax
}

pub struct NeuralNet {
    dim_output: usize,
    layers: Vec<Layer>,
    pub alpha: f64
}

impl NeuralNet {

    pub fn new(dim_output: usize) -> NeuralNet {
        // Create a new model with input dimensions dim_input
        
        NeuralNet{dim_output:dim_output, layers: Vec::new(), alpha: 0.05f64}
    }

    pub fn linear(&mut self, dim: usize) {
        let next_layer: Layer = Layer::Linear{
            bias: Matrix::<f64>::random::<f64>(1, dim),
            weights: Matrix::<f64>::random::<f64>(self.dim_output, dim),
            accumulated_err:  Matrix::<f64>::fill(0f64, 1, dim)
        };
        self.layers.push(next_layer);
        self.dim_output = dim;
    }

    pub fn sigmoid(&mut self, dim: usize) {
        let next_layer: Layer = Layer::Sigmoid{
            bias: Matrix::<f64>::random::<f64>(1, dim),
            weights: Matrix::<f64>::random::<f64>(self.dim_output, dim),
            accumulated_err:  Matrix::<f64>::fill(0f64, 1, dim)
        };
        self.layers.push(next_layer);
        self.dim_output = dim;
    }

    pub fn softmax(&mut self) {
        let next_layer: Layer = Layer::Softmax{};
        self.layers.push(next_layer);
    }



    pub fn forward_prop(& self, inp: &Matrix::<f64>) -> Matrix::<f64> {
        let mut x = inp.clone();
        for layer in self.layers.iter(){
            match layer {
                Layer::Linear{bias, weights, accumulated_err} => {
                    x = x.mul(&weights, false, false);
                    x.iadd(&bias);
                },
                Layer::Sigmoid{bias, weights, accumulated_err} =>{
                    x = x.mul(&weights, false, false);
                    x.iadd(&bias);
                    x.isigmoid();
                },
                Layer::Softmax{} =>{
                    let mut sm = Matrix::<f64>::fill(0f64, 1, self.dim_output);
                    let mut norm: f64 = 0f64;
                    for (smi, xi) in sm.iter_mut().zip(x.iter()){
                        let ex: f64 = 2.71828f64.powf(*xi);
                        *smi = ex;
                        norm = norm + ex;
                    }
                    sm.idiv_scalar(norm);
                    x = sm;
                },
                Layer::Dropout{reject} => {

                }
            }
        }
    // return x after all the transformations
    x
    }

    pub fn backprop(&mut self, inp: &Matrix::<f64>) -> f64 {
        // take inp as an input, perform forward prop and backprop, 
        // return the AVERAGE error at tne output layer
        let mut error: Matrix::<f64> = self.forward_prop(inp);
        0f64
    }
}


fn main() {

    let mut nn = NeuralNet::new(10); 
    nn.linear(20);
    nn.sigmoid(12);
    nn.sigmoid(4);
    nn.softmax();
    let mut input = Matrix::<f64>::random::<f64>(1,10);
    input.get(0, 1).replace(&22f64).unwrap();

    let out = nn.forward_prop(&input);
    println!("OUTPUT = {}", out);
    //println!("INOUT={}", input);
    

}

//let b = Matrix::<T>::fill(2f64, 300, 1_000);
//let a = mat![1.0, 2.0; 3.0, 4.0];