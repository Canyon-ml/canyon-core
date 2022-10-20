
// Import Module, gT
use super::*;

pub struct ELU{
    output: Tensor,
    delta: Tensor,
}

impl ELU {
    pub fn new4d (prev_size: (usize, usize, usize, usize)) -> Self {
        Self {
            output: Tensor::new(prev_size.0, prev_size.1, prev_size.2, prev_size.3),
            delta: Tensor::new(prev_size.0, prev_size.1, prev_size.2, prev_size.3),
        }
    }

    pub fn new2d (prev_size: (usize, usize)) -> Self {
        Self {
            output: Tensor::from_shape((prev_size.0, prev_size.1, 1, 1)),
            delta: Tensor::from_shape((prev_size.0, prev_size.1, 1, 1)),
        }
    }
}

impl Module<'_> for ELU {
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
                let exp = f32::exp(-i);
                *o = 0.1 * (exp - 1.0);
                *a = 0.1 * exp;
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