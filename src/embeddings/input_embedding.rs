use candle_core::{Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};

pub struct InputEmbedding {
    scale_factor: Tensor,
    embedding: Embedding,
}

impl InputEmbedding {
    pub fn new(vocab_size: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        println!("vocab_size: {}, d_model: {}", vocab_size, d_model);
        let device = vb.device();
        let scale_factor = Tensor::new((d_model as f32).sqrt(), device).unwrap();
        let embedding = embedding(vocab_size, d_model, vb.pp("wte")).unwrap();
        Ok(Self {
            scale_factor,
            embedding,
        })
    }
}

impl Module for InputEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // (batch_sz, seq_len) --> (batch_sz, seq_len, d_model)
        // Multiply by sqrt(d_model) to scale the embeddings according to the paper
        let output = self
            .embedding
            .forward(&xs)?
            .broadcast_mul(&self.scale_factor)?;
        // println!("InputEmbedding output: {:?}", output.dtype());
        Ok(output)
    }
}

// #[cfg(test)]
// mod tests {
//     use candle_core::DType;
//     use candle_nn::VarMap;
//     use tokenizers::tokenizer::Tokenizer;
//     use crate::utils::device;
//     use super::*;

//     #[test]
//     fn test_input_embedding() {
//         let device = device(true).unwrap();
//         let var_map = VarMap::new();
//         let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
//         let tokenizer = Tokenizer::from_pretrained("dustalov/wikitext-wordlevel", None).unwrap();
//         let vocab_size = tokenizer.get_vocab_size(true);
//         let input_embedding = InputEmbedding::new(vocab_size, 512, vb);
//         let input = "hello world, this is a test.";
//         let binding = tokenizer.encode(input, true).unwrap();
//         let token_ids = binding.get_ids();
//         let toekn_ids_len = token_ids.len();
//         println!("token_ids len: {:?}", toekn_ids_len);
//         let idx = Tensor::from_slice(token_ids, (toekn_ids_len, ), &device).unwrap();
//         let output = input_embedding.forward(&idx).unwrap();
//         println!("output dims: {:?}", output.shape());
//         // assert_eq!(output.shape(), &[toekn_ids_len, 512]);

//     }
// }
