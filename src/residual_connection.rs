use candle_core::{DType, Result, Tensor};
use candle_nn::{layer_norm, Dropout, LayerNormConfig, Module, VarBuilder};

use crate::layer_norm::LayerNormalization;

// def __init__(self, features: int, dropout: float) -> None:
// super().__init__()
// self.dropout = nn.Dropout(dropout)
// self.norm = LayerNormalization(features)

// def forward(self, x, sublayer):
// return x + self.dropout(sublayer(self.norm(x)))

pub struct ResidualConnection {
    dropout: Dropout,
    norm: LayerNormalization,
}

impl ResidualConnection {
    pub fn new(features: usize, drop_p: f32, vb: VarBuilder) -> Result<Self> {
        let dropout = Dropout::new(drop_p);
        // let norm = LayerNormalization(layer_norm(
        //     features,
        //     LayerNormConfig::default(),
        //     vb.pp("residual_norm"),
        // )?);
        let norm = LayerNormalization::new(features, 1e-5, vb.pp("residual_norm"))?;

        Ok(Self { dropout, norm })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        sub_layer: &dyn Fn(&Tensor) -> Result<Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        // println!("residual_connection input: {:?}", xs);
        let xs_dtype = xs.dtype();
        let internal_dtype = DType::F32;
        let xs = xs.to_dtype(internal_dtype)?;
        let output = (&xs
            + self
                .dropout
                .forward(&sub_layer(&self.norm.forward(&xs)?)?, train)?)?
        .to_dtype(xs_dtype)?;
        // println!("residual_connection output: {:?}", output.shape());
        Ok(output)
    }
}
