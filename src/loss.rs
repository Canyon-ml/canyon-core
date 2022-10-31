
use gt::Tensor;

pub enum Loss {
    /// Mean Absolute Error
    MAE,
    /// Mean Squared Error
    MSE,
    /// Categorical Cross Entropy
    CrossEntropy,
    /// Used for Classification
    Hinge,
    /// Typically used for regression
    Huber,
    /// ?
    KullbackLeibler,
    /// Root Mean Squared Error
    RMSE,
}

impl Loss {
    pub fn compute (&self, output: &Tensor, target: &Tensor, delta: &mut Tensor) -> f32 {

        let mut loss: f32 = 0.0;

        match self {

            // - Mean Absolute Error - //
            Loss::MAE => {
                for batch in 0..output.rows {
                    for col in 0..output.cols {
                        loss += f32::abs(output[(batch, col)] - target[(batch, col)]);
                        // compute the derivative into delta
                        // delta = output > target then 1.0, else -1.0
                        delta[(batch, col)] = if 
                            output[(batch, col)] > target[(batch, col)] 
                                { 1.0 } else { -1.0 };
                    }
                }
            },

            // - Mean Squared Error - //
            Loss::MSE => {
                for batch in 0..output.rows {
                    for col in 0..output.cols {
                        loss += f32::powi(output[(batch, col)] - target[(batch, col)], 2);
                        // delta = -2 * (target - output)
                        delta[(batch, col)] = -2.0 * (target[(batch, col)] - output[(batch, col)]);
                    }
                }
            },

            Loss::CrossEntropy => {
                for batch in 0..output.rows {
                    for col in 0..output.cols {
                        loss += -(target[(batch, col)] * f32::log10(output[(batch, col)]) + (1.0 - target[(batch, col)]) * f32::log10(1.0 - output[(batch, col)]));
                        
                        delta[(batch, col)] = (target[(batch, col)] - output[(batch, col)]);
                    }
                }
            },
            Loss::Hinge => todo!(),
            Loss::Huber => todo!(),
            Loss::KullbackLeibler => todo!(),
            Loss::RMSE => todo!(),
        }

        loss / output.len() as f32
    }
}