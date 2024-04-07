use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Dropout, Linear, Module, VarBuilder};

#[cfg(feature = "metal")]
use candle_core::D;
#[cfg(feature = "metal")]
use candle_nn::ops::softmax;

#[cfg(not(feature = "metal"))]
use candle_nn::ops::softmax_last_dim;

use crate::utils::masked_fill;

fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    dropout: Option<&Dropout>,
    train: bool,
) -> Result<(Tensor, Tensor)> {
    let head_size = q.dims().last().unwrap();
    let scale_factor = Tensor::new((*head_size as f32).sqrt(), q.device())?.to_dtype(q.dtype())?;

    // (batch_sz, num_heads, seq_len, seq_len) = (batch_sz, num_heads, seq_len, head_sz) * (batch_sz, num_heads, head_sz, seq_len)
    let attention_scores = q.matmul(&k.t()?)?.broadcast_div(&scale_factor)?;

    let attention_scores = match mask {
        Some(m) => masked_fill(&attention_scores, &m)?,
        None => attention_scores,
    };

    #[cfg(not(feature = "metal"))]
    let attention_weights = softmax_last_dim(&attention_scores)?;

    #[cfg(feature = "metal")]
    let attention_weights = softmax(&attention_scores, D::Minus1)?;

    let attention_weights = match dropout {
        Some(d) => d.forward(&attention_weights, train)?,
        None => attention_weights,
    };

    // (batch_sz, num_heads, seq_len, head_sz) = (batch_sz, num_heads, seq_len, seq_len) * (batch_sz, num_heads, seq_len, head_sz)
    let attention_output = attention_weights.matmul(&v)?;

    Ok((attention_output, attention_weights))
}

pub struct MultiHeadAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    dropout: Dropout,
    num_heads: usize,
    head_size: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize, drop_p: f32, vb: VarBuilder) -> Result<Self> {
        let w_q = linear_no_bias(d_model, d_model, vb.pp("w_q"))?;
        let w_k = linear_no_bias(d_model, d_model, vb.pp("w_k"))?;
        let w_v = linear_no_bias(d_model, d_model, vb.pp("w_v"))?;
        let w_o = linear_no_bias(d_model, d_model, vb.pp("w_o"))?;
        let dropout = Dropout::new(drop_p);
        let num_heads = num_heads;
        let head_size = d_model / num_heads;

        Ok(Self {
            w_q,
            w_k,
            w_v,
            w_o,
            dropout,
            num_heads,
            head_size,
        })
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, d_model) = query.dims3()?;
        let q = self.w_q.forward(query)?; // (batch_sz, seq_len, d_model) -> (batch_sz, seq_len, d_model)
        let k = self.w_k.forward(key)?; // (batch_sz, seq_len, d_model) -> (batch_sz, seq_len, d_model)
        let v = self.w_v.forward(value)?; // (batch_sz, seq_len, d_model) -> (batch_sz, seq_len, d_model)

        // println!("batch_size: {:?}, seq_len: {:?}, d_model: {:?}", batch_size, seq_len, d_model);
        let q = q
            .reshape((q.dims()[0], q.dims()[1], self.num_heads, self.head_size))?
            .permute((0, 2, 1, 3))?
            .contiguous()?; // (batch_sz, seq_len d_model) -> (batch_sz, num_heads, seq_len, head_sz)
        let k = k
            .reshape((k.dims()[0], k.dims()[1], self.num_heads, self.head_size))?
            .permute((0, 2, 1, 3))?
            .contiguous()?; // (batch_sz, seq_len d_model) -> (batch_sz, num_heads, seq_len, head_sz)
        let v = v
            .reshape((v.dims()[0], v.dims()[1], self.num_heads, self.head_size))?
            .permute((0, 2, 1, 3))?
            .contiguous()?; // (batch_sz, seq_len d_model) -> (batch_sz, num_heads, seq_len, head_sz)

        // ((batch_sz, num_heads, seq_len, head_sz), (batch_sz, num_heads, seq_len, seq_len))
        let (attention_output, attention_weights) =
            attention(&q, &k, &v, mask, Some(&self.dropout), train)?;

        let attention_output = attention_output // (batch_sz, num_heads, seq_len, head_sz)
            .transpose(1, 2)? // (batch_sz, seq_len, num_heads, head_sz)
            .contiguous()?
            .reshape((batch_size, seq_len, d_model))? // (batch_sz, seq_len, d_model)
            .apply(&self.w_o)?; // (batch_sz, seq_len, d_model)

        Ok(attention_output)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use crate::{multi_head_attention::attention, utils::causal_mask};
    #[test]
    fn test_attention() {
        let device = Device::Cpu;

        let q = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (1, 1, 1, 4), &device).unwrap();
        let k = q.clone();
        let v = q.clone();

        let mask = causal_mask(q.dims()[0], &device).unwrap();

        let (attention_scores, attention_weights) =
            attention(&q, &k, &v, Some(&mask), None, false).unwrap();
        println!("attention_scores: {:?}", attention_scores.dims());
        println!("attention_weights: {:?}", attention_weights.dims());

        assert_eq!(attention_scores.shape(), q.shape());
        assert!(attention_scores
            .flatten_all()
            .unwrap()
            .to_vec0::<f32>()
            .iter()
            .all(|&x| x >= 0.0f32 && x <= 1.0f32));
    }
}
