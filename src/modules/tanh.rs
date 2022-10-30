
// Import Module, gT
use super::*;

pub struct Tanh {
    output: Tensor,
    delta: Tensor,
}

impl Tanh {
    pub fn new (prev: (usize, usize, usize, usize)) -> Self {
        Self {
            output: Tensor::new(prev.0, prev.1, prev.2, prev.3),
            delta: Tensor::new(prev.0, prev.1, prev.2, prev.3),
        }
    }
}

impl Module<'_> for Tanh {
    fn forward <'a> (&mut self, input: &'a Tensor) -> &Tensor {
        
        // Compute the Sigmoid of the Input Tensor. ( 1 / 1 + exp^-i )
        // Also compute the Derivative. ( sigmoid(i) * (1 - simgoid(i)) )
        for ((i, o), d) in 
            input.iter().zip(self.output.iter_mut()).zip(self.delta.iter_mut()) {

            let top = f32::exp(*i) - f32::exp(-*i);
            let bot = f32::exp(*i) + f32::exp(-*i);
            *o = top / bot;
            *d = 1.0 - f32::powi(*o, 2);
        }

        &self.output
    }

    fn backward (&mut self, delta: &Tensor) -> &Tensor {
        
        // Use the pre-computed derivative to find the delta
        Tensor::mul(&mut self.delta, delta);

        &self.delta
    }
}