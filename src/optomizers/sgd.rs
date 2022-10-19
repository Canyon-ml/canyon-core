
use super::*;

pub struct SGD {
    lr: f32
}

impl SGD {
    pub fn new (lr: f32) -> Self {
        Self { lr }
    }
}

impl Optomizer for SGD {
    fn optomize (&mut self, weight: &mut Tensor, delta: &Tensor) {
        if weight.len() == delta.len() {
            Tensor::update(weight, delta, self.lr);
        } else {
            Tensor::update_bias(weight, delta, self.lr);
        }
    }
}