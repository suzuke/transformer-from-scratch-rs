use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, LayerNormConfig, ModuleT, VarBuilder};

use crate::{
    feed_forward::FeedForward, layer_norm::LayerNormalization,
    multi_head_attention::MultiHeadAttention, residual_connection::ResidualConnection,
};

pub struct DecoderBlock {
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    residual_connections: Vec<ResidualConnection>,
}

impl DecoderBlock {
    pub fn new(
        features: usize,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        drop_p: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut residual_connections = Vec::with_capacity(3);
        for i in 0..3 {
            residual_connections.push(ResidualConnection::new(
                features,
                drop_p,
                vb.pp(format!("residual_connection_{i}")),
            )?);
        }
        Ok(Self {
            self_attention,
            cross_attention,
            feed_forward,
            residual_connections,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        encoder_output: &Tensor,
        src_mask: &Tensor,
        tgt_mask: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let mut x = xs.clone();
        x = self.residual_connections[0].forward(
            &x,
            &|x| self.self_attention.forward(x, x, x, Some(tgt_mask), train),
            train,
        )?;
        x = self.residual_connections[1].forward(
            &x,
            &|x| {
                self.cross_attention.forward(
                    x,
                    encoder_output,
                    encoder_output,
                    Some(src_mask),
                    train,
                )
            },
            train,
        )?;
        x = self.residual_connections[2].forward(
            &x,
            &|x| self.feed_forward.forward_t(x, train),
            train,
        )?;
        Ok(x)
    }
}

pub struct Decoder {
    blocks: Vec<DecoderBlock>,
    norm: LayerNormalization,
}

impl Decoder {
    pub fn new(features: usize, blocks: Vec<DecoderBlock>, vb: VarBuilder) -> Result<Self> {
        // let norm = LayerNormalization(layer_norm(
        //     features,
        //     LayerNormConfig::default(),
        //     vb.pp("norm"),
        // )?);
        let norm = LayerNormalization::new(features, 1e-5, vb.pp("norm"))?;
        Ok(Self { blocks, norm })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        encoder_output: &Tensor,
        src_mask: &Tensor,
        tgt_mask: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let mut x = xs.clone();
        for block in &self.blocks {
            x = block.forward(&x, encoder_output, src_mask, tgt_mask, train)?;
        }
        x = self.norm.forward_t(&x, train)?;
        Ok(x)
    }
}
