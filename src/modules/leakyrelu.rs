
// Import Module, gT
use super::*;

pub struct LeakyReLU {
    output: Tensor,
    delta: Tensor,
}

impl LeakyReLU {
    pub fn new (prev: (usize, usize, usize, usize)) -> Self {
        Self {
            output: Tensor::new(prev.0, prev.1, prev.2, prev.3),
            delta: Tensor::new(prev.0, prev.1, prev.2, prev.3),
        }
    }
}

impl Module for LeakyReLU {
    fn forward  (&mut self, input: &Tensor) -> &Tensor {
        // I: Input
        // O: self.output
        // A: self.aprime
        // Compute the ReLU and its Derivative
        for ((i, o), a) in 
            input.iter().zip(self.output.iter_mut()).zip(self.delta.iter_mut()) {
            if *i > 0.0 { 
                *o = *i; *a = 1.0;
            } else {
                *o = 0.1 * *i; *a = 0.1;
            }
        }
        
        &self.output
    }

    fn backward (&mut self, delta: &Tensor) -> &Tensor {
        // Multiply the Derivative of this layer by Delta into the input
        Tensor::mul(&mut self.delta, delta);

        // Return the delta
        &self.delta
    }
}