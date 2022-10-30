
// Import Module, gT
use super::*;

pub struct MaxPool {
    /// The Output for this Layer
    output: Tensor,

    /// The Delta wrt the Input
    delta: Tensor,

    /// The Output of the Previous Layer
    input: Tensor,

    /// The Stride for the Pooling
    /// 0 is normal operation
    stride: usize,

    /// The height and width of the filter
    filter_size: usize,
}

impl MaxPool {
    pub fn new (prev: (usize, usize, usize, usize), filter_size: usize, stride: usize) -> Self {
        Self {
            output: Tensor::new(
                (prev.0 - (filter_size - 1)) / stride,
                (prev.1 - (filter_size - 1)) / stride,
                prev.2, prev.3
            ),
            delta: Tensor::new(prev.0, prev.1, prev.2, prev.3),
            input: TEMP.clone(), stride, filter_size,
        }
    }
}

impl Module for MaxPool {
    fn forward  (&mut self, input: &Tensor) -> &Tensor {
        // Cache the Input for use in the backward pass
        self.input = input.clone();

        // Perform the Max Pool Forward operation. 
        Tensor::max_pool_forward(input, &mut self.output, self.stride, self.filter_size);

        &self.output
    }

    fn backward (&mut self, delta: &Tensor) -> &Tensor {

        // Fill delta with zeros
        self.delta.fill(0.0);

        // Perform the Max Pool Backward Operation
        Tensor::max_pool_backward(&self.input, &self.output, delta, &mut self.delta, self.stride, self.filter_size);

        &self.delta
    }
}