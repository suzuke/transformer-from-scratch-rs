use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct ProjectionLayer {
    proj: Linear,
}

impl ProjectionLayer {
    pub fn new(d_model: usize, vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear(d_model, vocab_size, vb)?;
        Ok(Self { proj })
    }
}

impl Module for ProjectionLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        self.proj.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Shape, Tensor};
    use candle_nn::{Module, VarBuilder, VarMap};

    use super::ProjectionLayer;

    #[test]
    fn test_projection_layer() {
        let d_model = 512;
        let vocab_size = 10000;
        let device = Device::Cpu;
        let dtype = DType::F32;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let projection_layer = ProjectionLayer::new(d_model, vocab_size, vb).unwrap();

        let (batch_size, seq_len) = (1, 5);
        let xs = Tensor::ones((batch_size, seq_len, d_model), dtype, &device).unwrap();
        let output = projection_layer.forward(&xs).unwrap();

        assert_eq!(
            output.shape(),
            &Shape::from_dims(&[batch_size, seq_len, vocab_size])
        );
    }
}
