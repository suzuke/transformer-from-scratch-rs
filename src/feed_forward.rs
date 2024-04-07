use candle_core::{Result, Tensor};
use candle_nn::{linear, Activation, Dropout, Linear, ModuleT, VarBuilder};

pub struct FeedForward {
    linear1: Linear,
    dropout: Dropout,
    linear2: Linear,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, dropout: f32, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear(d_model, d_ff, vb.pp("ff_linear1"))?;
        let dropout = Dropout::new(dropout);
        let linear2 = linear(d_ff, d_model, vb.pp("ff_linear2"))?;

        Ok(Self {
            linear1,
            dropout,
            linear2,
        })
    }
}

impl ModuleT for FeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let output = xs
            .apply(&self.linear1)?
            .apply(&Activation::Relu)?
            .apply_t(&self.dropout, train)?
            .apply(&self.linear2)?;
        Ok(output)
    }
}
