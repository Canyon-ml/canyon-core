
// Import Module, gT
use super::*;

pub struct Sigmoid {
    output: Tensor,
    delta: Tensor,
}

impl Sigmoid {
    pub fn new (prev: (usize, usize, usize, usize)) -> Self {
        Self {
            output: Tensor::new(prev.0, prev.1, prev.2, prev.3),
            delta: Tensor::new(prev.0, prev.1, prev.2, prev.3),
        }
    }
}

impl Module for Sigmoid {
    fn forward  (&mut self, input: &Tensor) -> &Tensor {
        
        // Compute the Sigmoid of the Input Tensor. ( 1 / 1 + exp^-i )
        // Also compute the Derivative. ( sigmoid(i) * (1 - simgoid(i)) )
        for i in 0..input.len() {
            self.output[i] = 1.0 / (1.0 + f32::exp(-input[i]));
            self.delta[i] = self.output[i] * (1.0 - self.output[i]);
        }

        &self.output
    }

    fn backward (&mut self, delta: &Tensor) -> &Tensor {
        
        // Use the pre-computed derivative to find the delta
        Tensor::mul(&mut self.delta, delta);

        &self.delta
    }
}