
// Import Module, Optim, gT. 
use super::*;

pub struct Conv<'b> {
    /// The Output for this Layer.
    /// - rows: ((input.rows - padding * 2) - kernel.rows + 1) / stride
    /// - cols: ((input.cols - padding * 2) - kernel.cols + 1) / stride
    /// - channels: kernel.duration (number of filters)
    /// - duration: batch size
    output: Tensor,

    /// The Output of the Previous Layer
    input: &'b Tensor,

    /// The Kernel for this Layer.
    /// Contains multiple Filters.
    /// - rows: custom 
    /// - cols: custom 
    /// - channels: a.channels
    /// - duration: custom (number of filters)
    kernel: Tensor,

    /// The Bias of each Channel.
    /// - rows: output.channels
    /// - cols: 1
    /// - channels: 1
    /// - duration: 1
    /// 
    /// Each element of bias corresponds to 1 channel of the output. 
    /// That element is added to that entire channel. 
    bias: Tensor,

    /// The Delta with respect to the Kernel.
    /// Same size as the Kernel. 
    del_w: Tensor,

    /// The Delta with respect to the input
    del_i: Tensor,
    
    /// The Stride of the Kernel.
    /// How many cells it jumps for every step.
    /// Set to Zero for normal operation
    stride: usize,

    /// The Padding of the Kernel.
    /// The Amount of Zeroes placed around the border of the input Tensor.
    padding: usize,

    /// The Optomizer for this Layer
    optim: Optim,
}

impl Conv<'_> {
    pub fn new (
        prev_size: (usize, usize, usize), kernel_size: usize, num_filters: usize, 
        batch_size: usize, padding: usize, stride: usize, optim: Optim
    ) -> Self {
        Self {
            output: Tensor::new(
                ((prev_size.0 - padding * 2) - kernel_size + 1) / stride,
                ((prev_size.1 - padding * 2) - kernel_size + 1) / stride,
                kernel_size * kernel_size * prev_size.2, batch_size
            ),
            input: TEMP, 
            del_w: Tensor::new(kernel_size, kernel_size, prev_size.2, batch_size),
            del_i: Tensor::new(prev_size.0, prev_size.1, prev_size.2, batch_size),
            kernel: Tensor::new_random(kernel_size, kernel_size, prev_size.2, num_filters, (-0.3, 0.3)),
            bias: Tensor::new_random(1, 1, prev_size.2, 1, (-0.3, 0.3)),
            stride, padding, optim
        }
    }
}

impl<'b, 'a: 'b> Module<'a> for Conv<'b> {
    fn forward  (&mut self, input: &'a Tensor) -> &Tensor {
        // Set the self.input variable for use next time
        self.input = input;

        // Convolve across the Input with the Kernel into Output
        Tensor::convolve(&input, &mut self.output, &self.kernel, self.stride, self.padding, 1);

        // Add the bias channel-by-channel
        Tensor::add_bias_conv(&mut self.output, &self.bias);

        // Return our output to be used by the next layer
        &self.output
    }

    fn backward (&mut self, delta: &Tensor) -> &Tensor {
        // Optomize the bias with the delta wrt this layer
        self.optim.optomize(&mut self.bias, delta);

        // Compute the Gradient wrt the weights
        Tensor::convolve(&self.input, &mut self.del_w, delta, 1, self.padding, self.stride);

        // The Padding of the backward step wrt the inputs
        let back_pad = (((self.output.rows) + (self.output.rows - 1) * self.stride) - 1) + self.padding;

        // Compute the Gradient wrt the Input
        // Rotate the Kernel to be accurate with the _tr fn
        Tensor::convolve_tr(&self.kernel, &mut self.del_i, delta, 1, back_pad, self.stride);

        // Optomize the Weights
        // do it after the weights so we dont mess up the convolve_tr into input op
        self.optim.optomize(&mut self.kernel, &self.del_w);

        // Return del_i to be used by next layer
        &self.del_i
    }
}