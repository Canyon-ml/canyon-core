
use super::*;

pub struct Flatten {
    output: Tensor,
    delta: Tensor,
}

impl Flatten {
    pub fn new (prev: (usize, usize, usize, usize)) -> Self {
        Self {
            output: Tensor::new(prev.3, prev.0 * prev.1 * prev.2, 1, 1),
            delta: Tensor::new(prev.0, prev.1, prev.2, prev.3),
        }
    }
}

impl Module for Flatten {
    fn forward  (&mut self, input: &Tensor) -> &Tensor {
        
        // Copy data into flattened Tensor
        Tensor::copy(input, &mut self.output);

        &self.output
    }

    fn backward (&mut self, delta: &Tensor) -> &Tensor {
        
        // Copy data out of flattened tensor
        Tensor::copy(delta, &mut self.delta);

        &self.delta
    }
}