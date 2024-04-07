use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Result, Tensor};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!("Running on CPU, to run on GPU(metal), build with `--features metal`");
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub fn causal_mask(size: usize, device: &Device) -> Result<Tensor> {
    Tensor::tril2(size, DType::U8, device)?.unsqueeze(0)
}

pub fn masked_fill(attn_weights: &Tensor, attn_mask: &Tensor) -> Result<Tensor> {
    let attn_device = attn_weights.device();
    let attn_dtype = attn_weights.dtype();
    let attn_shape = attn_weights.shape();

    let attn_mask = attn_mask.broadcast_as(attn_shape)?;
    let mask_value = Tensor::new(f32::NEG_INFINITY, attn_device)?
        .broadcast_as(attn_shape)?
        .to_dtype(attn_dtype)?;
    attn_mask.where_cond(&attn_weights, &mask_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn test_masked_fill() {
        let device = Device::Cpu;

        let attn_weights = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let attn_mask = Tensor::from_vec(vec![0u32, 1, 0, 1], (1, 4), &device).unwrap();

        let result = masked_fill(&attn_weights, &attn_mask).unwrap();

        let expected = Tensor::from_vec(
            vec![f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY, 4.0],
            (1, 4),
            &device,
        )
        .unwrap();
        assert_eq!(
            result.to_vec2::<f32>().unwrap(),
            expected.to_vec2::<f32>().unwrap()
        );
    }

    #[test]
    fn test_causal_mask() {
        let size = 3;
        let device = Device::Cpu;

        let result = causal_mask(size, &device).unwrap();
        println!("result: {:?}", result.to_vec3::<u8>());
        assert_eq!(result.dims3().unwrap(), (1, size, size));
        // Further assertions for dtype and device checks...
        // Verify upper triangular property...
        assert!(result
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap()
            .iter()
            .all(|x| *x == 0u8 || *x == 1u8));

        let a = Tensor::arange(0u32, 3, &device)
            .unwrap()
            .reshape((1, 1, 3))
            .unwrap();
        let b = a.broadcast_as(&[1, 3, 3]).unwrap();
        println!("a: {:?}", a.to_vec3::<u32>());
        println!("b:{:?} {:?}", b.shape(), b.to_vec3::<u32>());
    }
}
