
extern crate gtensor as gt;

mod modules;
mod optomizers;
mod loss;

pub use optomizers::Optim;
pub use modules::*;
pub use loss::Loss;