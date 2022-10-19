
// - gTensor Includes - //
use gt::Tensor;

// - Local Includes - //
use crate::optomizers::Optim;

pub trait Module<'a> {
    fn forward  (&mut self, input: &'a Tensor) -> &Tensor;
    fn backward (&mut self, delta: &Tensor) -> &Tensor;
}

pub(crate) mod dense; // ID 0
pub(crate) mod conv;       // ID 1
pub(crate) mod sigmoid;    // ID 2
pub(crate) mod softmax;    // ID 3
pub(crate) mod relu;       // ID 4
pub(crate) mod tanh;       // ID 5
pub(crate) mod leakyrelu;  // ID 6  
pub(crate) mod elu;        // ID 7
pub(crate) mod max_pool;   // ID 8
pub(crate) mod flatten;    // ID 9

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
static TEMP: &Tensor = &Tensor { 
    data: Vec::new(), rows: 1, cols: 1, channels: 1, duration: 1, tag: 1 
};