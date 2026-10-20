//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use candle_core::{
    quantized::{QMatMul, QStorage, QTensor},
    DType, Device, Module, Result, Tensor,
};
use candle_nn::Linear;
use std::{borrow::Cow, sync::Arc};

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

/// Perform indexed MoE forward pass on a QTensor by dequantizing and using UnquantLinear.
///
/// # Arguments
/// * `qtensor` - The quantized weight tensor [num_experts, n, k]
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn qtensor_indexed_moe_forward(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    // Repacked per-expert gemv path; falls back to dequantize-and-gather only for
    // layouts the packed kernels cannot serve. Normalize the metal/cpu 4D/5D input
    // shapes to the (tokens, x_t, hidden) form the kernel expects.
    {
        let (x3, ids2, out_shape): (Tensor, Tensor, Option<Vec<usize>>) = match *x.dims() {
            [b, s, xt, h] => {
                let (ib, is, t) = ids.dims3()?;
                if ib == b && is == s {
                    (
                        x.reshape((b * s, xt, h))?,
                        ids.reshape((b * s, t))?,
                        Some(vec![b, s, t, 0]),
                    )
                } else {
                    (x.clone(), ids.clone(), None)
                }
            }
            [b, s, 1, 1, h] => {
                let (ib, is, t) = ids.dims3()?;
                if ib == b && is == s {
                    (
                        x.reshape((b * s, 1, h))?,
                        ids.reshape((b * s, t))?,
                        Some(vec![b, s, t, 0]),
                    )
                } else {
                    (x.clone(), ids.clone(), None)
                }
            }
            [_, _, _] => (x.clone(), ids.clone(), Some(vec![])),
            _ => (x.clone(), ids.clone(), None),
        };
        if let Some(shape) = out_shape.filter(|_| x3.rank() == 3 && ids2.rank() == 2) {
            let out = match qtensor.indexed_gemv(&x3, &ids2)? {
                Some(out) => Some(out),
                None if x3.device().is_cpu() => Some(routed_qmatmul(qtensor, &x3, &ids2)?),
                None => None,
            };
            if let Some(out) = out {
                return if shape.is_empty() {
                    Ok(out)
                } else {
                    let n_out = out.dim(2)?;
                    out.reshape((shape[0], shape[1], shape[2], n_out))
                };
            }
        }
    }

    let device = x.device();

    // Dequantize all weights to f32
    let weights = qtensor.dequantize(device)?;

    // Create an UnquantLinear and use its gather_forward
    let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weights, None)))?;

    unquant.gather_forward(x, ids)
}

/// Quantized matmul of each routed expert on its own slice of the bank, instead of dequantizing every expert.
fn routed_qmatmul(qtensor: &QTensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (num_experts, n, k) = qtensor.shape().dims3()?;
    let (tokens, _, _) = x.dims3()?;
    let (_, topk) = ids.dims2()?;
    let rows = x
        .to_dtype(DType::F32)?
        .broadcast_as((tokens, topk, k))?
        .reshape((tokens * topk, k))?;
    let data = qtensor.data()?;
    let expert_bytes = data.len() / num_experts;
    let out = crate::unquantized::grouped_expert_matmul(
        &rows,
        &ids.flatten_all()?,
        num_experts,
        |e, xs| {
            let bytes = &data[e * expert_bytes..(e + 1) * expert_bytes];
            let storage = QStorage::from_data(Cow::Borrowed(bytes), &Device::Cpu, qtensor.dtype())?;
            QMatMul::from_qtensor(QTensor::new(storage, (n, k))?)?.forward(xs)
        },
    )?;
    out.reshape((tokens, topk, n))
}

/// Perform indexed MoE forward pass on a QMatMul.
///
/// This is the main entry point for CPU/Metal GGUF quantized MoE forward.
///
/// # Arguments
/// * `qmatmul` - The quantized weight matrix
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn cpu_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, x, ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            // For non-quantized tensors, use UnquantLinear directly
            let unquant =
                UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(t.clone(), None)))?;
            unquant.gather_forward(x, ids)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    #[test]
    fn routed_qmatmul_matches_dequantized_gather() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::randn(0f32, 1., (6, 16, 64), &device)?;
        let qtensor = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let x = Tensor::randn(0f32, 1., (5, 1, 64), &device)?;
        let ids = Tensor::new(&[[0u32, 3], [5, 3], [1, 0], [3, 2], [4, 4]], &device)?;

        let dequant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            qtensor.dequantize(&device)?,
            None,
        )))?;
        let reference = dequant.gather_forward(&x, &ids)?;
        let routed = routed_qmatmul(&qtensor, &x, &ids)?;

        // activations get quantized to q8 in the quantized matmul, so allow a little slack
        let err = (reference.clone() - routed)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let scale = reference.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 0.02 * scale, "err {err} scale {scale}");
        Ok(())
    }
}
