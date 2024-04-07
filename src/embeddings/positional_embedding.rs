use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Dropout, ModuleT, VarBuilder};

pub struct PositionalEmbedding {
    embedding: Tensor,
    dropout: Dropout,
}

impl PositionalEmbedding {
    pub fn new(
        max_position_embeddings: usize,
        d_model: usize,
        drop_p: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device();
        let dtype = vb.dtype();

        let dropout = Dropout::new(drop_p);

        let half_d_model = d_model / 2;

        // 1 / 10_000^(2i / d_model)
        // = 10_000^(-2i / d_model)
        // = exp{ ln[ 10_000^(-2i / d_model) ] }
        // = exp{ -2i / d_model * ln[ 10_000 ] }
        let div_term = (Tensor::arange(0., half_d_model as f32, device)?
            * (-2. * (10_000f64.ln() / d_model as f64)))?
            .exp()?
            .to_dtype(dtype)?
            .reshape((1, half_d_model))?;

        let pos = Tensor::arange(0., max_position_embeddings as f32, device)?
            .to_dtype(dtype)?
            .reshape((max_position_embeddings, 1))?;

        let pe_base = pos.matmul(&div_term)?;
        let pe_sin = pe_base.sin()?;
        let pe_cos = pe_base.cos()?;

        // Interleaving sin and cos
        // Stack along a new dimension (creates an extra dimension)
        let stacked = Tensor::stack(&[&pe_sin, &pe_cos], 2)?;
        // Now, we have a shape (max_position_embeddings, d_model, 2)

        // Reshape to interleave sin and cos values
        let embedding = stacked
            .reshape((max_position_embeddings, d_model))? // (max_position_embeddings, d_model)
            .contiguous()?
            .unsqueeze(0)?; // (1, max_position_embeddings, d_model)

        Ok(Self { dropout, embedding })
    }
}

impl ModuleT for PositionalEmbedding {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        // println!("PositionalEmbedding forward");
        // xs is of shape (batch_size, seq_len, d_model)
        let (_, seq_len, _) = xs.dims3()?;

        // TODO: check xs shape
        // println!("xs shape: {:?}", xs.shape());
        let xs = xs.broadcast_add(&self.embedding.i((.., ..seq_len, ..))?)?; // no grad here?

        let output = self.dropout.forward(&xs, train)?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::DType;
    use candle_nn::VarMap;

    use crate::utils::device;

    use super::*;

    const DTYPE: DType = DType::F64;

    #[test]
    fn test_positional_embedding() {
        let max_position_embeddings = 10;
        let d_model = 512;

        let device = &device(true).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DTYPE, device);

        let pos_emb = PositionalEmbedding::new(max_position_embeddings, d_model, 0.1, vb).unwrap();

        let (batch_size, embedding_size, out_dim) = pos_emb.embedding.dims3().unwrap();
        assert_eq!(
            (batch_size, max_position_embeddings, d_model),
            (1, embedding_size, out_dim)
        );

        let (batch_size, seq_len) = (1, 5);
        let xs = Tensor::ones((batch_size, seq_len, d_model), DTYPE, device).unwrap();
        let output = pos_emb.forward_t(&xs, true).unwrap();
        assert_eq!(output.dims3().unwrap(), (batch_size, seq_len, d_model));
    }

    #[test]
    fn test_embedding_values_within_expected_range() {
        let max_position_embeddings = 10;
        let d_model = 512;

        let device = &device(true).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DTYPE, device);
        let pos_emb = PositionalEmbedding::new(max_position_embeddings, d_model, 0.1, vb).unwrap();

        let embeddings = pos_emb.embedding.to_vec3::<f64>().unwrap();
        assert!(embeddings
            .iter()
            .flatten()
            .into_iter()
            .flatten()
            .all(|x| *x >= -1. && *x <= 1.));
    }
}
