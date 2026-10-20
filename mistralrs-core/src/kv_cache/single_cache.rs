use candle_core::{DType, Result, Tensor, D};

use super::{quant, NormalCache};

#[derive(Debug, Clone)]
pub struct SingleCacheSnapshot {
    pub current_seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct SingleCache {
    // all_data is an option on a Tensor, this makes it possible to only create the actual tensor
    // on the first call where the batch size is easily known.
    // Also this makes it safe to clone a KvCache that has been reset (as in it will not share
    // its internal state with the cloned instance).
    pub all_data: Option<Tensor>,
    pub dim: usize,
    pub current_seq_len: usize,
    pub capacity_seq_len: usize,
    pub max_seq_len: usize,
    // dtype handed back from a quantized (u8) cache
    pub out_dtype: Option<DType>,
}

impl SingleCache {
    pub fn new(dim: usize, max_seq_len: usize, capacity_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            max_seq_len,
            capacity_seq_len,
            out_dtype: None,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> Option<&Tensor> {
        self.all_data.as_ref()
    }

    pub fn current_data(&self) -> Result<Option<Tensor>> {
        match self.current_raw()? {
            Some(d) if d.dtype() == DType::U8 => Ok(Some(quant::dequantize(
                &d,
                self.out_dtype.unwrap_or(DType::F32),
            )?)),
            d => Ok(d),
        }
    }

    /// Stored data without dequantizing, for moving caches between sequences.
    pub fn current_raw(&self) -> Result<Option<Tensor>> {
        self.all_data
            .as_ref()
            .map(|d| d.narrow(self.dim, 0, self.current_seq_len))
            .transpose()
    }

    pub fn snapshot(&self) -> SingleCacheSnapshot {
        SingleCacheSnapshot {
            current_seq_len: self.current_seq_len,
        }
    }

    pub fn can_append_from_snapshot(
        &self,
        snapshot: &SingleCacheSnapshot,
        append_len: usize,
    ) -> bool {
        snapshot.current_seq_len == self.current_seq_len
            && snapshot
                .current_seq_len
                .checked_add(append_len)
                .is_some_and(|len| len <= self.max_seq_len)
    }

    pub fn rollback_to(&mut self, keep_len: usize) -> candle_core::Result<()> {
        self.set_len(keep_len)
    }

    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.all_data = None;
    }

    pub fn try_set_len(&self, len: usize) -> candle_core::Result<()> {
        if len > self.capacity_seq_len {
            candle_core::bail!(
                "kv-cache: requested length ({}) exceeds current capacity ({})",
                len,
                self.capacity_seq_len
            );
        }
        Ok(())
    }

    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        self.try_set_len(len)?;
        self.current_seq_len = len;
        Ok(())
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let quantized;
        let src = if quant::packed_dim(src.dim(D::Minus1)?).is_some() {
            self.out_dtype = Some(src.dtype());
            quantized = quant::quantize(src)?;
            &quantized
        } else {
            src
        };
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad);
        };

        // Expand kv cache
        if self.current_seq_len + seq_len > self.capacity_seq_len {
            let diff = self.current_seq_len + seq_len - self.capacity_seq_len;
            let n_blocks_needed = diff.div_ceil(NormalCache::CACHE_GROW_SIZE);
            self.capacity_seq_len += n_blocks_needed * NormalCache::CACHE_GROW_SIZE;
            if self.capacity_seq_len > self.max_seq_len {
                candle_core::bail!(
                    "kv-cache: requested capacity ({}) above max seq len ({})",
                    self.capacity_seq_len,
                    self.max_seq_len
                )
            }
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            ad.slice_set(self.all_data.as_ref().unwrap(), self.dim, 0)?;
            self.all_data = Some(ad);
        }

        let ad = self.all_data.as_mut().unwrap();

        ad.slice_set(src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}
