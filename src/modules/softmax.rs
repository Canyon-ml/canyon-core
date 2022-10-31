
// Import Module, gT. 
use super::*;

/// Always follows a Fully Connected Layer, never a Conv Layer.
/// Used for Multi-Class Classification.
pub struct Softmax {
    output: Tensor,
    delta: Tensor,
}

impl Softmax {
    pub fn new (prev: (usize, usize, usize, usize)) -> Self {
        Self {
            output: Tensor::new(prev.0, prev.1, prev.2, prev.3),
            delta: Tensor::new(prev.0, prev.1, prev.2, prev.3),
        }
    }
}

impl Module for Softmax {
    fn forward  (&mut self, input: &Tensor) -> &Tensor {
        for batch in 0..input.rows {
            let mut sum: f32 = 0.0;
            for col in 0..input.cols {
                self.output[(batch, col)] = f32::exp(input[(batch, col)]);
                sum += self.output[(batch, col)];
            }

            for col in 0..input.cols {
                self.output[(batch, col)] = self.output[(batch, col)] / sum;
                self.delta[(batch, col)] = self.output[(batch, col)] * (1.0 - self.output[(batch, col)]);
            }
        }

        &self.output
    }

    fn backward (&mut self, delta: &Tensor) -> &Tensor {
        // Multiply the Derivative of this layer by Delta
        Tensor::mul(&mut self.delta, delta);

        // Return the delta
        &self.delta
    }
}