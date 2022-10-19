
// - gTensor Includes - //
use gt::Tensor;

pub type Optim = Box<dyn Optomizer>;

pub trait Optomizer {
    fn optomize (&mut self, weight: &mut Tensor, delta: &Tensor);
}

pub(crate) mod sgd;

pub use sgd::SGD;