
use std::rc::Rc;

// - gTensor Includes - //
use gt::Tensor;

// - Local Includes - //
use crate::optomizers::Optim;

pub trait Module {
    /// The Forward Pass of a Module
    /// Computes the output of the Module
    fn forward  (&mut self, input: &Tensor) -> &Tensor;

    /// The Backward Pass of a Module
    /// Computes the gradient of the Module
    /// If this layer has them, update the trainable params. 
    fn backward (&mut self, delta: &Tensor) -> &Tensor;
}

pub(crate) mod dense;      
pub(crate) mod conv;       
pub(crate) mod sigmoid;    
pub(crate) mod softmax;    
pub(crate) mod relu;       
pub(crate) mod tanh;       
pub(crate) mod leakyrelu;    
pub(crate) mod elu;        
pub(crate) mod max_pool;   
pub(crate) mod flatten;    

pub use dense::Dense;
pub use conv::Conv;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use relu::ReLU;
pub use tanh::Tanh;
pub use leakyrelu::LeakyReLU;
pub use elu::ELU;
pub use max_pool::MaxPool;
pub use flatten::Flatten;

/// A Dummy Variable used to initialize empty references
static TEMP: Tensor = Tensor { 
    data: Vec::new(), rows: 1, cols: 1, channels: 1, duration: 1
};