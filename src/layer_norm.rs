use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Init, LayerNorm, Module, VarBuilder};

// #[derive(Clone, Debug)]
// pub struct LayerNormalization(pub LayerNorm);

// impl Module for LayerNormalization {
//     fn forward(&self, xs: &Tensor) -> Result<Tensor> {
//         //not yet implemented: no unary function for u32, use F32 instead.
//         let xs = xs.to_dtype(DType::F32)?;
//         self.0.forward(&xs)
//     }
// }

// #[cfg(test)]
// mod tests {
//     use candle_core::{DType, Device, Tensor};
//     use candle_nn::{layer_norm, LayerNormConfig, Module, VarBuilder, VarMap};

//     use super::LayerNormalization;

//     #[test]
//     fn test_layer_norm() {
//         let device = Device::Cpu;
//         let dtype = DType::F32;
//         let varmap = VarMap::new();
//         let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
//         let input_tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap();
//         let layer_norm = LayerNormalization(layer_norm(4, LayerNormConfig::default(), vb).unwrap());
//         let output = layer_norm.forward(&input_tensor).unwrap();

//         let output = output.to_vec1::<f32>().unwrap();

//         //check rms of output is close to 1
//         let rms = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
//         println!("rms: {}", rms);
//         assert!(rms - 1.0 < f32::EPSILON);

//         //check mean of output is close to 0
//         let mean = output.iter().sum::<f32>() / output.len() as f32;
//         println!("mean: {}", mean);
//         assert!(mean < f32::EPSILON);
//     }
// }

// class LayerNormalization(nn.Module):

//     def __init__(self, features: int, eps:float=10**-6) -> None:
//         super().__init__()
//         self.eps = eps
//         self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
//         self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

//     def forward(self, x):
//         # x: (batch, seq_len, hidden_size)
//          # Keep the dimension for broadcasting
//         mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
//         # Keep the dimension for broadcasting
//         std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
//         # eps is to prevent dividing by zero or when std is very small
//         return self.alpha * (x - mean) / (std + self.eps) + self.bias

pub struct LayerNormalization {
    eps: f64,
    alpha: Tensor,
    bias: Tensor,
}

impl LayerNormalization {
    pub fn new(features: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get_with_hints(features, "alpha", Init::Const(1.))?;
        let bias = vb.get_with_hints(features, "bias", Init::Const(0.))?;
        Ok(Self { eps, alpha, bias })
    }
}

impl Module for LayerNormalization {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = xs.mean_keepdim(D::Minus1)?;
        let std_sqr = xs.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let result = xs
            .broadcast_sub(&mean)?
            .broadcast_div(&(std_sqr + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.alpha)?
            .broadcast_add(&self.bias)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{test_utils::to_vec3_round, Device};
    use candle_nn::VarMap;

    use super::*;

    #[test]
    fn test_layer_norm() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let layer = LayerNormalization::new(3, 1e-5, vb).unwrap();
        let xs = Tensor::new(
            &[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]],
            &Device::Cpu,
        )
        .unwrap();
        let ys = layer.forward(&xs).unwrap();
        println!("ys: {:?}", to_vec3_round(&ys, 4).unwrap());
        assert_eq!(
            to_vec3_round(&ys, 4).unwrap(),
            &[[
                [-1.2247, 0.0, 1.2247],
                [-1.2247, 0.0, 1.2247],
                [1.2247, 0.0, -1.2247]
            ]]
        );
    }
}
