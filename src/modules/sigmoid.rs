
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
        for ((i, o), d) in 
            input.iter().zip(self.output.iter_mut()).zip(self.delta.iter_mut()) {

            *o = 1.0 / 1.0 + f32::exp(-*i);
            *d = *o * (1.0 - *o);
        }

        #[cfg(feature = "debug")]
        println!("output of Sigmoid: {:?}", self.output.data);

        &self.output
    }

    fn backward (&mut self, delta: &Tensor) -> &Tensor {
        
        // Use the pre-computed derivative to find the delta
        Tensor::mul(&mut self.delta, delta);

        &self.delta
    }
}