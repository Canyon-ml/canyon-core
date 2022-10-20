
use super::*;

pub struct Flatten {
    output: Tensor,
    delta: Tensor,
}

impl Flatten {
    pub fn new (prev_size: (usize, usize, usize), batch_size: usize) -> Self {
        Self {
            output: Tensor::new(batch_size, prev_size.0 * prev_size.1 * prev_size.2, 1, 1),
            delta: Tensor::new(prev_size.0, prev_size.1, prev_size.2, batch_size),
        }
    }
}

impl Module<'_> for Flatten {
    fn forward <'a> (&mut self, input: &'a Tensor) -> &Tensor {
        
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