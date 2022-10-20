
// Import Module, gT. 
use super::*;

/// Always follows a Fully Connected Layer, never a Conv Layer.
/// Used for Multi-Class Classification.
pub struct Softmax {
    output: Tensor,
    delta: Tensor,
}

impl Softmax {
    pub fn new (prev_size: (usize, usize)) -> Self {
        Self {
            output: Tensor::from_shape((prev_size.0, prev_size.1, 1, 1)),
            delta: Tensor::from_shape((prev_size.0, prev_size.1, 1, 1)),
        }
    }
}

impl Module<'_> for Softmax {
    fn forward <'a> (&mut self, input: &'a Tensor) -> &Tensor {
        for batch in 0..input.rows {
            let mut sum: f32 = 0.0;
            for col in 0..input.cols {
                sum += input[(batch, col)]
            }

            for col in 0..input.cols {
                self.output[(batch, col)] = input[(batch, col)] / sum;
                self.delta[(batch, col)]= self.output[(batch, col)] * (1.0 - self.output[(batch, col)]);
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