use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use candle_core::{DType, Device, DeviceLocation, Result, Tensor, D};

pub const KV_QUANT_ENV: &str = "MISTRALRS_KV_QUANT";
const Q8_MAX: f64 = 127.;
const Q4_MAX: f64 = 7.;
const GROUP: usize = 32;
// 1 / sqrt(GROUP), the Hadamard normalization
const HADAMARD_NORM: f32 = 0.176_776_7;
// scale bytes hold ceil(log2(scale) * EXP_STEPS) + EXP_OFFSET as a little-endian u16
const EXP_STEPS: f64 = 256.;
const EXP_OFFSET: f64 = 32768.;
const SCALE_BYTES: usize = 2;
const MIN_ABSMAX: f64 = 1e-30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvQuant {
    Q8,
    Q4,
}

pub fn kv_quant() -> Option<KvQuant> {
    static MODE: OnceLock<Option<KvQuant>> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var(KV_QUANT_ENV).as_deref() {
            Ok("q8") => Some(KvQuant::Q8),
            Ok("q4") => Some(KvQuant::Q4),
            Ok(other) => {
                tracing::warn!("Ignoring {KV_QUANT_ENV}={other}, expected `q8` or `q4`.");
                None
            }
            Err(_) => None,
        };
        if let Some(mode) = mode {
            tracing::info!("Quantizing the non-paged KV cache to {mode:?}.");
        }
        mode
    })
}

impl KvQuant {
    fn qmax(self) -> f64 {
        match self {
            Self::Q8 => Q8_MAX,
            Self::Q4 => Q4_MAX,
        }
    }

    fn data_bytes(self, head_dim: usize) -> usize {
        match self {
            Self::Q8 => head_dim,
            Self::Q4 => head_dim / 2,
        }
    }
}

/// Bytes per stored row for a head dim, or None when that head dim stays unquantized.
pub fn packed_dim(head_dim: usize) -> Option<usize> {
    packed_dim_as(head_dim, kv_quant()?)
}

fn packed_dim_as(head_dim: usize, mode: KvQuant) -> Option<usize> {
    head_dim
        .is_multiple_of(GROUP)
        .then(|| mode.data_bytes(head_dim) + SCALE_BYTES * head_dim / GROUP)
}

fn head_dim(packed: usize, mode: KvQuant) -> usize {
    let per_group = mode.data_bytes(GROUP) + SCALE_BYTES;
    packed / per_group * GROUP
}

// normalized Sylvester Hadamard: symmetric and orthogonal, so it is its own inverse
fn hadamard(device: &Device) -> Result<Tensor> {
    static CACHE: OnceLock<Mutex<HashMap<DeviceLocation, Tensor>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(h) = cache.get(&device.location()) {
        return Ok(h.clone());
    }
    let data: Vec<f32> = (0..GROUP * GROUP)
        .map(|i| {
            let (r, c) = (i / GROUP, i % GROUP);
            if (r & c).count_ones() % 2 == 0 {
                HADAMARD_NORM
            } else {
                -HADAMARD_NORM
            }
        })
        .collect();
    let h = Tensor::from_vec(data, (GROUP, GROUP), device)?;
    cache.insert(device.location(), h.clone());
    Ok(h)
}

/// Returns (scale as f32, scale encoded as u8 bytes with a trailing dim of SCALE_BYTES).
fn encode_scale(absmax: &Tensor, qmax: f64) -> Result<(Tensor, Tensor)> {
    let ln2 = std::f64::consts::LN_2;
    let e = absmax
        .maximum(MIN_ABSMAX)?
        .affine(1. / qmax, 0.)?
        .log()?
        .affine(EXP_STEPS / ln2, 0.)?
        .ceil()?
        .affine(1., EXP_OFFSET)?
        .clamp(0., u16::MAX as f64)?;
    let hi = e.affine(1. / 256., 0.)?.floor()?;
    let lo = (&e - hi.affine(256., 0.)?)?;
    let bytes = Tensor::cat(&[lo, hi], D::Minus1)?.to_dtype(DType::U8)?;
    Ok((decode_exp(&e)?, bytes))
}

fn decode_exp(e: &Tensor) -> Result<Tensor> {
    e.affine(
        std::f64::consts::LN_2 / EXP_STEPS,
        -EXP_OFFSET * std::f64::consts::LN_2 / EXP_STEPS,
    )?
    .exp()
}

fn decode_scale(bytes: &Tensor) -> Result<Tensor> {
    let b = bytes.to_dtype(DType::F32)?;
    let e = (b.narrow(D::Minus1, 0, 1)? + b.narrow(D::Minus1, 1, 1)?.affine(256., 0.)?)?;
    decode_exp(&e)
}

/// Packs `[.., head_dim]` into u8 rows of `packed_dim(head_dim)` bytes.
pub fn quantize(x: &Tensor) -> Result<Tensor> {
    quantize_as(
        x,
        kv_quant().expect("quantize called without a KV quant mode"),
    )
}

fn quantize_as(x: &Tensor, mode: KvQuant) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let lead = &dims[..dims.len() - 1];
    let d = *dims.last().unwrap();
    let groups = d / GROUP;
    // rotating each group spreads outlier channels out, so one channel no longer sets the whole group's scale
    let x = x
        .to_dtype(DType::F32)?
        .reshape([lead, &[groups, GROUP]].concat())?
        .broadcast_matmul(&hadamard(x.device())?)?;
    let (scale, scale_bytes) = encode_scale(&x.abs()?.max_keepdim(D::Minus1)?, mode.qmax())?;
    let q = x
        .broadcast_div(&scale)?
        .round()?
        .clamp(-mode.qmax(), mode.qmax())?
        .affine(1., mode.qmax() + 1.)?;
    let data = match mode {
        KvQuant::Q8 => q.reshape([lead, &[d]].concat())?,
        KvQuant::Q4 => {
            let q = q.reshape([lead, &[d / 2, 2]].concat())?;
            (q.narrow(D::Minus1, 0, 1)? + q.narrow(D::Minus1, 1, 1)?.affine(16., 0.)?)?
                .squeeze(D::Minus1)?
        }
    };
    Tensor::cat(
        &[
            data.to_dtype(DType::U8)?,
            scale_bytes.reshape([lead, &[groups * SCALE_BYTES]].concat())?,
        ],
        D::Minus1,
    )
}

/// Inverse of [`quantize`].
pub fn dequantize(x: &Tensor, dtype: DType) -> Result<Tensor> {
    dequantize_as(
        x,
        kv_quant().expect("u8 KV cache without a KV quant mode"),
        dtype,
    )
}

fn dequantize_as(x: &Tensor, mode: KvQuant, dtype: DType) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let lead = &dims[..dims.len() - 1];
    let d = head_dim(*dims.last().unwrap(), mode);
    let groups = d / GROUP;
    let data_bytes = mode.data_bytes(d);
    let b = x.narrow(D::Minus1, 0, data_bytes)?.to_dtype(DType::F32)?;
    let q = match mode {
        KvQuant::Q8 => b,
        KvQuant::Q4 => {
            let hi = b.affine(1. / 16., 0.)?.floor()?;
            let lo = (&b - hi.affine(16., 0.)?)?;
            Tensor::stack(&[lo, hi], D::Minus1)?
        }
    };
    let scale = decode_scale(
        &x.narrow(D::Minus1, data_bytes, groups * SCALE_BYTES)?
            .reshape([lead, &[groups, SCALE_BYTES]].concat())?,
    )?;
    q.reshape([lead, &[groups, GROUP]].concat())?
        .affine(1., -(mode.qmax() + 1.))?
        .broadcast_mul(&scale)?
        .broadcast_matmul(&hadamard(x.device())?)?
        .reshape([lead, &[d]].concat())?
        .to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    // keys with a few large outlier channels, the case that breaks naive 4-bit KV
    fn keys(dev: &Device) -> Result<Tensor> {
        let x = Tensor::randn(0f32, 1., (2, 4, 33, 128), dev)?;
        let outliers = Tensor::from_vec(
            (0..128)
                .map(|c| if c % 37 == 5 { 20f32 } else { 1. })
                .collect::<Vec<_>>(),
            128,
            dev,
        )?;
        x.broadcast_mul(&outliers)
    }

    fn rel_err(a: &Tensor, b: &Tensor) -> Result<f32> {
        let num = (a - b)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let den = a.sqr()?.sum_all()?.to_scalar::<f32>()?;
        Ok((num / den).sqrt())
    }

    #[test]
    fn hadamard_is_its_own_inverse() -> Result<()> {
        let h = hadamard(&Device::Cpu)?;
        let eye = Tensor::eye(GROUP, DType::F32, &Device::Cpu)?;
        assert!(rel_err(&eye, &h.matmul(&h)?)? < 1e-5);
        Ok(())
    }

    #[test]
    fn roundtrip_error_and_size() -> Result<()> {
        let dev = Device::Cpu;
        let x = keys(&dev)?;
        for (mode, packed, max_err) in [(KvQuant::Q8, 136, 0.01), (KvQuant::Q4, 72, 0.15)] {
            let q = quantize_as(&x, mode)?;
            assert_eq!(q.dtype(), DType::U8);
            assert_eq!(q.dims(), &[2, 4, 33, packed]);
            let y = dequantize_as(&q, mode, DType::F32)?;
            let err = rel_err(&x, &y)?;
            assert!(err < max_err, "{mode:?} relative error {err}");
        }
        Ok(())
    }

    #[test]
    fn q4_rotation_beats_plain_groups_on_outliers() -> Result<()> {
        let x = keys(&Device::Cpu)?;
        let y = dequantize_as(&quantize_as(&x, KvQuant::Q4)?, KvQuant::Q4, DType::F32)?;
        let rotated = rel_err(&x, &y)?;
        // same 4-bit grouping without the rotation
        let g = x.reshape((2, 4, 33, 4, GROUP))?;
        let (scale, _) = encode_scale(&g.abs()?.max_keepdim(D::Minus1)?, Q4_MAX)?;
        let plain = g
            .broadcast_div(&scale)?
            .round()?
            .clamp(-Q4_MAX, Q4_MAX)?
            .broadcast_mul(&scale)?
            .reshape(x.shape())?;
        assert!(rotated < rel_err(&x, &plain)?);
        Ok(())
    }

    #[test]
    fn zero_rows_and_narrowed_views() -> Result<()> {
        let x = Tensor::zeros((1, 2, 8, 64), DType::BF16, &Device::Cpu)?;
        for mode in [KvQuant::Q8, KvQuant::Q4] {
            let q = quantize_as(&x, mode)?;
            let y = dequantize_as(&q.narrow(2, 0, 5)?, mode, DType::BF16)?;
            assert_eq!(y.dims(), &[1, 2, 5, 64]);
            assert_eq!(y.dtype(), DType::BF16);
            assert_eq!(
                y.abs()?
                    .sum_all()?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?,
                0.
            );
        }
        Ok(())
    }
}
