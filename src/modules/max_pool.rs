
// Import Module, gT
use super::*;

pub struct MaxPool<'b> {
    /// The Output for this Layer
    output: Tensor,

    /// The Delta wrt the Input
    delta: Tensor,

    /// The Output of the Previous Layer
    input: &'b Tensor,

    /// The Stride for the Pooling
    /// 0 is normal operation
    stride: usize,

    /// The height and width of the filter
    filter_size: usize,
}

impl MaxPool<'_> {
    pub fn new (prev_size: (usize, usize, usize), filter_size: usize, stride: usize, batch_size: usize) -> Self {
        Self {
            output: Tensor::new(
                (prev_size.0 - (filter_size - 1)) / stride,
                (prev_size.1 - (filter_size - 1)) / stride,
                prev_size.2, batch_size, 50
            ),
            delta: Tensor::new(prev_size.0, prev_size.1, prev_size.2, batch_size, 51),
            input: TEMP, stride, filter_size,
        }
    }
}

impl<'b, 'a: 'b> Module<'a> for MaxPool<'b> {
    fn forward (&mut self, input: &'a Tensor) -> &Tensor {
        // Cache the Input for use in the backward pass
        self.input = input;

        // Perform the Max Pool Forward operation. 
        Tensor::max_pool_forward(input, &mut self.output, self.stride, self.filter_size);

        &self.output
    }

    fn backward (&mut self, delta: &Tensor) -> &Tensor {

        // Perform the Max Pool Backward Operation
        Tensor::max_pool_backward(self.input, &self.output, delta, &mut self.delta, self.stride, self.filter_size);

        &self.delta
    }
}