use candle_core::{Result, Tensor};
use candle_nn::{Module, ModuleT, VarBuilder};

use crate::{
    decoder::{Decoder, DecoderBlock},
    embeddings::{input_embedding::InputEmbedding, positional_embedding::PositionalEmbedding},
    encoder::{Encoder, EncoderBlock},
    feed_forward::FeedForward,
    multi_head_attention::MultiHeadAttention,
    projection_layer::ProjectionLayer,
};

pub struct Transformer {
    encoder: Encoder,
    decoder: Decoder,
    src_embed: InputEmbedding,
    tgt_embed: InputEmbedding,
    src_pos_embed: PositionalEmbedding,
    tgt_pos_embed: PositionalEmbedding,
    projection_layer: ProjectionLayer,
}

impl Transformer {
    pub fn new(
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos_embed: PositionalEmbedding,
        tgt_pos_embed: PositionalEmbedding,
        projection_layer: ProjectionLayer,
    ) -> Result<Self> {
        Ok(Self {
            encoder,
            decoder,
            src_embed,
            tgt_embed,
            src_pos_embed,
            tgt_pos_embed,
            projection_layer,
        })
    }

    pub fn encode(&self, src: &Tensor, src_mask: &Tensor, train: bool) -> Result<Tensor> {
        let output = self.encoder.forward(
            &self
                .src_pos_embed
                .forward_t(&self.src_embed.forward(src)?, train)?,
            src_mask,
            train,
        )?;
        Ok(output)
    }

    pub fn decode(
        &self,
        encoder_output: &Tensor,
        src_mask: &Tensor,
        tgt: &Tensor,
        tgt_mask: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let output = self.decoder.forward(
            &self
                .tgt_pos_embed
                .forward_t(&self.tgt_embed.forward(tgt)?, train)?,
            encoder_output,
            src_mask,
            tgt_mask,
            train,
        )?;
        Ok(output)
    }

    pub fn project(&self, xs: &Tensor) -> Result<Tensor> {
        self.projection_layer.forward(xs)
    }
}

pub fn transformer(
    src_vocab_size: usize,
    tgt_vocab_size: usize,
    src_seq_len: usize,
    tgt_seq_len: usize,
    d_model: usize,
    n: usize,
    num_heads: usize,
    drop_p: f32,
    d_ff: usize,
    vb: VarBuilder,
) -> Result<Transformer> {
    let src_embed = InputEmbedding::new(src_vocab_size, d_model, vb.pp("src_embed"))?;
    let tgt_embed = InputEmbedding::new(tgt_vocab_size, d_model, vb.pp("tgt_embed"))?;

    let src_pos_embed =
        PositionalEmbedding::new(src_seq_len, d_model, drop_p, vb.pp("src_pos_embed"))?;
    let tgt_pos_embed =
        PositionalEmbedding::new(tgt_seq_len, d_model, drop_p, vb.pp("tgt_pos_embed"))?;

    let encoder_blocks = (0..n)
        .map(|i| {
            let self_attention = MultiHeadAttention::new(
                d_model,
                num_heads,
                drop_p,
                vb.pp(format!("encode_self_attention_{i}")),
            )
            .unwrap();
            let feed_forward = FeedForward::new(
                d_model,
                d_ff,
                drop_p,
                vb.pp(format!("encode_feed_forward_{i}")),
            )
            .unwrap();
            let encoder_block = EncoderBlock::new(
                d_model,
                self_attention,
                feed_forward,
                drop_p,
                vb.pp(format!("encode_block_{i}")),
            )
            .unwrap();
            encoder_block
        })
        .collect();

    let decoder_blocks = (0..n)
        .map(|i| {
            let self_attention = MultiHeadAttention::new(
                d_model,
                num_heads,
                drop_p,
                vb.pp(format!("decode_self_attention_{i}")),
            )
            .unwrap();
            let cross_attention = MultiHeadAttention::new(
                d_model,
                num_heads,
                drop_p,
                vb.pp(format!("decode_cross_attention_{i}")),
            )
            .unwrap();
            let feed_forward = FeedForward::new(
                d_model,
                d_ff,
                drop_p,
                vb.pp(format!("decode_feed_forward_{i}")),
            )
            .unwrap();
            let decoder_block = DecoderBlock::new(
                d_model,
                self_attention,
                cross_attention,
                feed_forward,
                drop_p,
                vb.pp(format!("decode_block_{i}")),
            )
            .unwrap();
            decoder_block
        })
        .collect();

    let encoder = Encoder::new(d_model, encoder_blocks, vb.pp("encoder"))?;
    let decoder = Decoder::new(d_model, decoder_blocks, vb.pp("decoder"))?;

    let projection_layer =
        ProjectionLayer::new(d_model, tgt_vocab_size, vb.pp("projection_layer"))?;

    let transformer = Transformer::new(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        src_pos_embed,
        tgt_pos_embed,
        projection_layer,
    )?;

    Ok(transformer)
}

// #[cfg(test)]
// mod tests {
//     use candle_core::{DType, Device, Tensor};
//     use candle_nn::{VarBuilder, VarMap};

//     use crate::utils::causal_mask;

//     use super::transformer;

//     #[test]
//     fn test_transformer() {
//         let features = 32;
//         let blocks = 2;
//         let seq_len = 10;
//         let d_model = 64;
//         let device = &Device::Cpu;
//         let dtype = DType::F32;
//         let varmap = VarMap::new();
//         let vb = VarBuilder::from_varmap(&varmap, dtype, device);
//         let transformer = transformer(
//             features, features, seq_len, seq_len, d_model, blocks, 1, 0.1, 16, vb,
//         )
//         .unwrap();

//         // let (batch_size, seq_len) = (1, 5);
//         // let xs = Tensor::ones((batch_size, seq_len, features), DType::U32, device).unwrap();
//         // let xs_mask = causal_mask(xs.dims()[1], dtype, device).unwrap().unsqueeze(0).unwrap();
//         // println!("xs_mask: {:?}", xs_mask.shape());
//         // let encode_xs = transformer.encode(&xs, &xs_mask, false).unwrap();
//         // println!("encode_xs: {:?}", encode_xs.shape());
//     }
// }
