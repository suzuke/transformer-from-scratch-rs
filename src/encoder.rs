use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, LayerNormConfig, ModuleT, VarBuilder};

use crate::{
    feed_forward::FeedForward, layer_norm::LayerNormalization,
    multi_head_attention::MultiHeadAttention, residual_connection::ResidualConnection,
};

pub struct EncoderBlock {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    residual_connections: Vec<ResidualConnection>,
}

impl EncoderBlock {
    pub fn new(
        features: usize,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        drop_p: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut residual_connections = Vec::with_capacity(2);
        residual_connections.push(ResidualConnection::new(
            features,
            drop_p,
            vb.pp("residual_connection_0"),
        )?);
        residual_connections.push(ResidualConnection::new(
            features,
            drop_p,
            vb.pp("residual_connection_1"),
        )?);

        Ok(Self {
            self_attention,
            feed_forward,
            residual_connections,
        })
    }

    pub fn forward(&self, xs: &Tensor, src_mask: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.residual_connections[0].forward(
            xs,
            &|xs| {
                self.self_attention
                    .forward(xs, xs, xs, Some(src_mask), train)
            },
            train,
        )?;
        let xs = self.residual_connections[1].forward(
            &xs,
            &|xs| self.feed_forward.forward_t(xs, train),
            train,
        )?;

        Ok(xs)
    }
}

pub struct Encoder {
    blocks: Vec<EncoderBlock>,
    norm: LayerNormalization,
}

impl Encoder {
    pub fn new(features: usize, blocks: Vec<EncoderBlock>, vb: VarBuilder) -> Result<Self> {
        // let norm = LayerNormalization(layer_norm(
        //     features,
        //     LayerNormConfig::default(),
        //     vb.pp("encoder_norm"),
        // )?);
        let norm = LayerNormalization::new(features, 1e-5, vb.pp("encoder_norm"))?;
        Ok(Self { blocks, norm })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = xs.clone();
        for block in &self.blocks {
            xs = block.forward(&xs, mask, train)?;
        }
        let output = self.norm.forward_t(&xs, train)?;
        Ok(output)
    }
}

// #[cfg(test)]
// mod tests {

//     #[test]
//     fn test_encoder() {
//         let features = 32;
//     }
// }
