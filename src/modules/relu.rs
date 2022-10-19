
// Import Module, gT
use super::*;

pub struct ReLU {
    output: Tensor,
    delta: Tensor,
}

impl ReLU {
    pub fn new4d (prev_size: (usize, usize, usize), batch_size: usize) -> Self {
        Self {
            output: Tensor::new(prev_size.0, prev_size.1, prev_size.2, batch_size, 30),
            delta: Tensor::new(prev_size.0, prev_size.1, prev_size.2, batch_size, 31),
        }
    }

    pub fn new2d (prev_size: (usize, usize)) -> Self {
        Self {
            output: Tensor::new(prev_size.0, prev_size.1, 1, 1, 30),
            delta: Tensor::new(prev_size.0, prev_size.1, 1, 1, 31),
        }
    }
}

impl Module<'_> for ReLU {
    fn forward <'a> (&mut self, input: &'a Tensor) -> &Tensor {
        // I: Input
        // O: self.output
        // A: self.aprime
        // Compute the ReLU and its Derivative
        for ((i, o), a) in 
            input.iter().zip(self.output.iter_mut()).zip(self.delta.iter_mut()) {
            if *i > 0.0 { 
                *o = *i; *a = 1.0;
            } else {
                *o = 0.0; *a = 0.0;
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