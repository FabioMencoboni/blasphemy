
//#[macro_use]
//extern crate rustml;

// https://www.youtube.com/watch?v=ueO_Ph0Pyqk
// https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
// loss for softmax = -log(predict@correct class) = sum (yjlogy^ j)
// error at output for softmax = <pred> - <actual>
use std::cmp;
pub use rustml::Matrix;
pub use rustml::ops::{MatrixMatrixOps,MatrixScalarOps};
pub use rustml::ops_inplace::{FunctionsInPlace,MatrixScalarOpsInPlace,MatrixMatrixOpsInPlace};

/// ```
/// use blasphemy::{NeuralNet, Matrix};
/// 
/// let mut nn = NeuralNet::new(4); 
/// nn.linear(5);
/// nn.linear(5);
/// nn.linear(3);
/// nn.softmax();
///
/// let mut err = 0f64;
/// for iter in 0..200{
///     let input = Matrix::from_vec(vec![1f64,0f64,0f64,1f64], 1,4);
///     let output = Matrix::from_vec(vec![1f64,0f64,0f64], 1,3);
///     err = nn.backprop(&input, &output);
///     let input = Matrix::from_vec(vec![0f64,2f64,2f64,0f64], 1,4);
///     let output = Matrix::from_vec(vec![0f64,1f64,0f64], 1,3);
///     err = nn.backprop(&input, &output);
///     if iter % 3 == 0 {
///         nn.grad_descent();
///         println!("iter={} error={}", iter, err);
///     }
///     
///     
/// }
/// let input = Matrix::from_vec(vec![1f64,0f64,0f64,1f64], 1,4);
/// let prediction = nn.forward_prop(&input);
/// ```

struct Linear {
    inp: Matrix::<f64>,
    z: Matrix::<f64>, // input
    bias: Matrix::<f64>,
    bias_err: Matrix::<f64>,
    weights: Matrix::<f64>,
    weights_err: Matrix::<f64>
}
struct Residual {
    inp_1: Matrix::<f64>,
    z_1: Matrix::<f64>, // input
    bias_1: Matrix::<f64>,
    bias_err_1: Matrix::<f64>,
    weights_1: Matrix::<f64>,
    weights_err_1: Matrix::<f64>,
    inp_2: Matrix::<f64>,
    z_2: Matrix::<f64>, // input
    bias_2: Matrix::<f64>,
    bias_err_2: Matrix::<f64>,
    weights_2: Matrix::<f64>,
    weights_err_2: Matrix::<f64>
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


fn linear_deriv(a: &Matrix::<f64>) -> Matrix::<f64>{
    // the derivative of the activation is a*(1-a)
    let mut g_prime = a.clone(); // d/dx of the activation functions
    for gi in g_prime.iter_mut() {
        if *gi < 0f64{
            *gi = 0f64;
        } else {
            *gi = 1f64;
        }
    }
    g_prime 
}
fn sigmoid_deriv(a: &Matrix::<f64>) -> Matrix::<f64>{
    // the derivative of the activation is a*(1-a)
    let mut g_prime = a.clone(); // d/dx of the activation functions
    for gi in g_prime.iter_mut() {
        *gi = *gi * (1f64 - *gi);
        if *gi < 0.05f64 {
            *gi = 0.05f64; // chop to help avoid vanishing gradients
        }
    }
    g_prime 
}


fn rand_init(cols:usize, rows: usize) -> Matrix::<f64>{
    let mut weights = Matrix::<f64>::random::<f64>(cols, rows);
        for wi in weights.iter_mut(){
            *wi = *wi - 0.5f64; //random produces numbers between 0-1
        }
    weights

}

pub fn clip(x: &mut Matrix::<f64>, min_val: f64, max_val: f64) {
    for  xi in x.iter_mut(){
        //let orig: f64 = xi.clone();
        //*xi = cmp::min(cmp::max(*xi, -100f64), 100f64);
        if *xi < min_val{
            *xi = min_val;
        }
        if *xi > max_val{
            *xi = max_val;
        }
    }
}
impl NeuralNet {

    pub fn new(dim_last: usize) -> NeuralNet {
        // Create a new model with input dimensions dim_input
        
        NeuralNet{dim_last:dim_last, layers: Vec::new(), alpha: 0.05f64}
    }

    pub fn linear(&mut self, dim: usize) {
        //weights.idiv_scalar(100f64);
        let next_layer: Layer = Layer::Linear{
            inp: Matrix::<f64>::fill(0f64, 1, self.dim_last),
            z: Matrix::<f64>::fill(0f64, 1, dim),
            bias: rand_init(1, dim),
            bias_err:  Matrix::<f64>::fill(0f64, 1, dim),
            weights: rand_init(self.dim_last, dim),
            weights_err:  Matrix::<f64>::fill(0f64, self.dim_last, dim)
        };
        self.layers.push(next_layer);
        self.dim_last = dim;
    }

    pub fn sigmoid(&mut self, dim: usize) {

        let next_layer: Layer = Layer::Sigmoid{
            inp: Matrix::<f64>::fill(0f64, 1, self.dim_last),
            z: Matrix::<f64>::fill(0f64, 1, dim),
            a: Matrix::<f64>::fill(0f64, 1, dim),
            bias: rand_init(1, dim),
            bias_err:  Matrix::<f64>::fill(0f64, 1, dim),
            weights: rand_init(self.dim_last, dim),
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
                    //clip(&mut x, 0f64, 100f64);
                    *z = x.clone();
                },
                Layer::Sigmoid{inp, z, a, bias, bias_err, weights, weights_err} =>{
                    *inp = x.clone();
                    x = x.mul(&weights, false, false);
                    x.iadd(&bias);
                    *z = x.clone();
                    for xi in x.iter_mut() {
                        *xi = 1f64/(1f64+2.718f64.powf(-*xi));
                    }
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
            //clip(&mut x, -100f64, 100f64);
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
                    bias_err.iadd(&error);
                    let weights_delta = inp.mul(&error, true, false);

                    weights_err.iadd(&weights_delta);

                    error = error.mul(&weights, false, true);
                    let g_prime = sigmoid_deriv(&inp);

                    error.imule(&g_prime);
                    
                }
                Layer::Linear{inp, z,bias,bias_err, weights,weights_err} => {
                    bias_err.iadd(&error);
                    let weights_delta = inp.mul(&error, true, false);

                    weights_err.iadd(&weights_delta);

                    error = error.mul(&weights, false, true);
                    let g_prime = linear_deriv(&inp);

                    error.imule(&g_prime);

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
                    *weights_err = weights_err.mul_scalar(-0.05f64);
                    clip(weights_err, -0.05f64, 0.05f64);
                    weights.iadd(weights_err);
                    weights.mul_scalar(0.98f64); // regularization
                    //println!("{}", &weights);
                    //println!("linear weights {}", &weights);
                    *weights_err = weights_err.mul_scalar(0f64);
                    *bias_err = bias_err.mul_scalar(-0.05f64);
                    clip(bias_err, -0.10f64, 0.10f64);
                    bias.iadd(bias_err);
                    //clip(bias, -100f64, 100f64);
                    
                    *bias_err = bias_err.mul_scalar(0f64);
                    //clip(weights, -100f64, 100f64);
                }
                Layer::Linear{inp, z, bias, bias_err, weights,weights_err} => {
                    *weights_err = weights_err.mul_scalar(-0.05f64);
                    clip(weights_err, -0.05f64, 0.05f64);
                    weights.iadd(weights_err);
                    weights.mul_scalar(0.98f64); // regularization
                    //println!("{}", &weights);
                    //println!("linear weights {}", &weights);
                    *weights_err = weights_err.mul_scalar(0f64);
                    *bias_err = bias_err.mul_scalar(-0.05f64);
                    clip(bias_err, -0.10f64, 0.10f64);
                    bias.iadd(bias_err);
                    //clip(bias, -100f64, 100f64);
                    
                    *bias_err = bias_err.mul_scalar(0f64);
                    //clip(weights, -100f64, 100f64);
                    
                }
                Layer::Dropout{reject} => ()
            }
            
        }

    }
}







#[test]
fn test_1() {

    let mut nn = NeuralNet::new(4); 
    nn.linear(5);
    nn.linear(7);
    nn.linear(5);
    nn.linear(3); 
    nn.softmax();

    let mut err: f64 = 0f64;
    for iter in 0..200{
        let input = Matrix::from_vec(vec![1f64,0f64,0f64,1f64], 1,4);
        let output = Matrix::from_vec(vec![1f64,0f64,0f64], 1,3);
        err = nn.backprop(&input, &output);
        let input = Matrix::from_vec(vec![0f64,2f64,2f64,0f64], 1,4);
        let output = Matrix::from_vec(vec![0f64,1f64,0f64], 1,3);
        err = nn.backprop(&input, &output);
        if iter % 3 == 0 {
            nn.grad_descent();
            println!("iter={} error={}", iter, err);
        }
    }
    assert!(err < 0.01f64);
    let input = Matrix::from_vec(vec![1f64,0f64,0f64,1f64], 1,4);
    let prediction = nn.forward_prop(&input);
    println!("prediction: {}", &prediction);
}

