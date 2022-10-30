
// Imports Module, Optim, and gT. 
use super::*;

/// A Fully Connected Layer is a regular layer
/// It performs O = I dot W + B
pub struct Dense<'b> {
    /// Holds the output of this layer in the Forward Step
    /// Found by the matrix multiplication of input x weight
    /// Batch Size x Layer Size.
    output: Tensor,

    /// The Weights of the Connections to this layers' Neurons
    /// Prev Size x This Size
    weight: Tensor,

    /// The Bias of each Neuron
    /// 1 x This Size
    bias: Tensor,

    /// The Output of the previous layer
    /// Stored here as a reference because we use it in the forward
    /// pass and the backward pass for computing the gradient. 
    input: &'b Tensor,

    /// The Gradient with respect to the Weight
    /// Same size as Weight
    del_w: Tensor,

    /// The Gradient with respect to the Input
    /// Batch Size x Prev Size
    del_i: Tensor,

    /// The Optomizer for this Layer
    optim: Optim,
}

impl Dense<'_> {
    /// - prev: (batch_size, prev_size) a.k.a. size of input.
    /// - size: The number of Neurons in this layer
    /// - optim: The Optomizer for this layer
    pub fn new (prev: (usize, usize), size: usize, optim: Optim) -> Self {
        Self {
            output: Tensor::new2d(prev.0, size),
            input: TEMP, // tag 1 - dummy variable for initializing empty references
            del_w: Tensor::new2d(prev.1, size),
            del_i: Tensor::new2d(prev.0, prev.1),
            weight: Tensor::new2d(prev.1, size),
            bias: Tensor::new2d(1, size),
            optim
        }
    }
}

impl<'b, 'a: 'b> Module<'a> for Dense<'b> {
    fn forward (&mut self, input: &'a Tensor) -> &Tensor {
        // Set the self.input variable for use next time
        self.input = input;

        // MatMul the Input Matrix by the Weight Matrix to produce Output
        // (For each Neuron compute a line of n dimensions)
        Tensor::multiply(false, input, false, &self.weight, &mut self.output);

        // Add the Bias to the Output Matrix
        // For each Neuron, add the base of the line computed.
        Tensor::add(&mut self.output, &self.bias);

        // Return our output so it can be used by the next layer.
        &self.output
    }

    fn backward (&mut self, delta: &Tensor) -> &Tensor {
        // Optomize the bias with the fresh delta
        self.optim.optomize(&mut self.bias, delta);

        // Compute the Gradient with respect to the Weights - resulting del_w is the same size as the weight tensor
        Tensor::multiply(true,&self.input, false, delta, &mut self.del_w);

        // Compute the Gradient with respect to the Input - Resulting Delta is the same size as previous layer
        Tensor::multiply(false, delta, true,&mut self.weight, &mut self.del_i);

        // Pass the weight tensor and delta for it to the optomizer
        self.optim.optomize(&mut self.weight, &self.del_w);

        // Return del_i to be used as delta in next module
        &self.del_i
    }
}