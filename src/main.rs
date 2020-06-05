
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
///     dim_last: usize,
///     layers: Vec<Layer>,
///     pub alpha: f64
/// }
/// ```
struct Linear {
    inp: Matrix::<f64>,
    z: Matrix::<f64>, // input
    bias: Matrix::<f64>,
    bias_err: Matrix::<f64>,
    weights: Matrix::<f64>,
    weights_err: Matrix::<f64>
}
struct Sigmoid {
    inp: Matrix::<f64>,
    z: Matrix<f64>,     // input = activation(input)
    a: Matrix::<f64>, // activation(input)
    bias: Matrix::<f64>,
    bias_err: Matrix::<f64>,
    weights: Matrix::<f64>,
    weights_err: Matrix::<f64>
}
struct Softmax{
}
struct Dropout{
    reject: f64
}

pub enum Layer {
    Linear{inp: Matrix::<f64>,z: Matrix::<f64>, bias: Matrix::<f64>, bias_err: Matrix::<f64>, weights:Matrix::<f64>,  weights_err: Matrix::<f64>},
    Sigmoid{inp: Matrix::<f64>,z: Matrix::<f64>,a: Matrix::<f64>, bias: Matrix::<f64>, bias_err: Matrix::<f64>, weights:Matrix::<f64>, weights_err: Matrix::<f64>},
    Dropout{reject:f64},
    Softmax
}

pub struct NeuralNet {
    dim_last: usize,
    layers: Vec<Layer>,
    pub alpha: f64
}


fn activation_deriv(a: &Matrix::<f64>) -> Matrix::<f64>{
    // the derivative of the activation is a*(1-a)
    let g_prime = a.clone(); // d/dx of the activation functions
    g_prime.mul_scalar(-1f64).add_scalar(1f64).imule(a);
    g_prime 
}

impl NeuralNet {

    pub fn new(dim_last: usize) -> NeuralNet {
        // Create a new model with input dimensions dim_input
        
        NeuralNet{dim_last:dim_last, layers: Vec::new(), alpha: 0.05f64}
    }

    pub fn linear(&mut self, dim: usize) {
        let mut weights = Matrix::<f64>::random::<f64>(self.dim_last, dim);
        weights.idiv_scalar(100f64);
        let next_layer: Layer = Layer::Linear{
            inp: Matrix::<f64>::fill(0f64, 1, self.dim_last),
            z: Matrix::<f64>::fill(0f64, 1, dim),
            bias: Matrix::<f64>::random::<f64>(1, dim),
            bias_err:  Matrix::<f64>::fill(0f64, 1, dim),
            weights: weights,
            weights_err:  Matrix::<f64>::fill(0f64, self.dim_last, dim)
        };
        self.layers.push(next_layer);
        self.dim_last = dim;
    }

    pub fn sigmoid(&mut self, dim: usize) {
        let mut weights = Matrix::<f64>::random::<f64>(self.dim_last, dim);
        weights.idiv_scalar(100f64);
        let next_layer: Layer = Layer::Sigmoid{
            inp: Matrix::<f64>::fill(0f64, 1, self.dim_last),
            z: Matrix::<f64>::fill(0f64, 1, dim),
            a: Matrix::<f64>::fill(0f64, 1, dim),
            bias: Matrix::<f64>::random::<f64>(1, dim),
            bias_err:  Matrix::<f64>::fill(0f64, 1, dim),
            weights: weights,
            weights_err:  Matrix::<f64>::fill(0f64, self.dim_last, dim)
        };
        self.layers.push(next_layer);
        self.dim_last = dim;
    }

    pub fn softmax(&mut self) {
        let next_layer: Layer = Layer::Softmax{};
        self.layers.push(next_layer);
    }



    pub fn forward_prop(&mut self, x_in: &Matrix::<f64>) -> Matrix::<f64> {
        let mut x = x_in.clone();
        for layer in self.layers.iter_mut(){
            match layer {
                Layer::Linear{inp, z, bias, bias_err, weights, weights_err} => {
                    *inp = x.clone();
                    x = x.mul(&weights, false, false);
                    x.iadd(&bias);
                    *z = x.clone();
                },
                Layer::Sigmoid{inp, z, a, bias, bias_err, weights, weights_err} =>{
                    *inp = x.clone();
                    x = x.mul(&weights, false, false);
                    x.iadd(&bias);
                    *z = x.clone();
                    x.isigmoid();
                    *a = x.clone();
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

    pub fn backprop(&mut self, x_in: &Matrix::<f64>, y_out: &Matrix::<f64>) -> f64 {
        // take inp as an input, perform forward prop and backprop, 
        // return the AVERAGE error at tne output layer
        let mut error: Matrix::<f64> = self.forward_prop(x_in);
        error.isub(y_out);
        //println!("{}", &error);
        let mut sum_error = 0f64;
        for ei in error.iter(){
            sum_error = sum_error + (ei*ei);
        }
        for layer in self.layers.iter_mut().rev(){
            match layer {
                Layer::Softmax{} => (),
                Layer::Sigmoid{inp, z, a, bias,bias_err, weights, weights_err} => {
                    //println!("Shp Error Inbound: {}", &error);
                    //println!("Shp Activation: {}", &activation);
                    //println!("bias_err{}", &bias_err);
                    //println!("error {}", &error);
                    bias_err.iadd(&error);
                    let weights_delta = inp.mul(&error, true, false);
                    //println!("weights_err {}", &weights_err);
                    //println!("weights_delta {}", &weights_delta);
                    weights_err.iadd(&weights_delta);

                    error = error.mul(&weights, false, true);
                    let g_prime = activation_deriv(&inp);
                    //println!("error {}", &error);
                    //println!("g_prime {}", &g_prime);
                    error.imule(&g_prime);

                    

                    
                }
                Layer::Linear{inp, z,bias,bias_err, weights,weights_err} => {

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
                Layer::Sigmoid{inp, z, a, bias, bias_err, weights, weights_err} => {
                    *weights_err = weights_err.mul_scalar(-0.12f64);
                    //println!("er,{}", weights_err);
                    weights.iadd(weights_err);
                    *weights_err = weights_err.mul_scalar(0f64);
                    //println!("WT,{}", weights);
                    *bias_err = bias_err.mul_scalar(-0.12f64);
                    bias.iadd(bias_err);
                    //println!("BIAS{}", &bias);
                    *bias_err = bias_err.mul_scalar(0f64);
                }
                Layer::Linear{inp, z, bias, bias_err, weights,weights_err} => {
                    *weights_err = weights_err.mul_scalar(-0.12f64);
                    weights.iadd(weights_err);
                    *weights_err = weights_err.mul_scalar(0f64);
                    *bias_err = bias_err.mul_scalar(-0.12f64);
                    bias.iadd(bias_err);
                    
                    *bias_err = bias_err.mul_scalar(0f64);
                    
                }
                Layer::Dropout{reject} => ()
            }
            
        }

    }
}


fn main() {

    let mut nn = NeuralNet::new(4); 
    //nn.linear(20);
    nn.sigmoid(5);
    nn.sigmoid(3);
    //nn.softmax();

    for iter in 0..10{
        let input = Matrix::from_vec(vec![1f64,0f64,0f64,1f64], 1,4);
        let output = Matrix::from_vec(vec![1f64,0f64,0f64], 1,3);
        let err = nn.backprop(&input, &output);
        nn.grad_descent();
        println!("iter={} error={}", iter, err);
    }
    //println!("OUTPUT = {}", out);
    //println!("INOUT={}", input);
    

}

//let b = Matrix::<T>::fill(2f64, 300, 1_000);
//let a = mat![1.0, 2.0; 3.0, 4.0];