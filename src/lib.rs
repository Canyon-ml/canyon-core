
pub extern crate gtensor as gt;

mod modules;
pub mod optomizers;
mod loss;
use modules::*;

pub use loss::Loss;

/// Enum Wrapper for Module Structs.
/// Makes it easy to interface and match
/// against different modules.
pub enum Mod {
    Dense (Dense),
    Conv (Conv),
    MaxPool (MaxPool),
    Flatten (Flatten),
    Sigmoid (Sigmoid),
    Softmax (Softmax),
    LeakyReLU (LeakyReLU),
    Tanh (Tanh),
    ReLU (ReLU),
    ELU (ELU),
}