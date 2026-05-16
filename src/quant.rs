use tracing::{debug};
use serde::{Serialize, Deserialize};

/// Fixed-point scaling factor. Q8.8 means 8 bits for integer part, 8 bits for fractional part.
/// Scaling factor = 2^8 = 256.
pub const DEFAULT_SCALE: f32 = 256.0;

/// Q8_0 scaling factor (e.g., 64.0 for Q2.6)
pub const I8_SCALE: f32 = 64.0;

pub struct Quantizer;

impl Quantizer {
    /// Converts an f32 to i16 using specified scaling and saturating cast.
    #[inline(always)]
    pub fn to_i16_scaled(val: f32, scale: f32) -> i16 {
        (val * scale).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }

    #[inline(always)]
    pub fn to_i16(val: f32) -> i16 {
        Self::to_i16_scaled(val, DEFAULT_SCALE)
    }

    /// Converts an i16 back to f32.
    #[inline(always)]
    pub fn to_f32_scaled(val: i16, scale: f32) -> f32 {
        val as f32 / scale
    }

    #[inline(always)]
    pub fn to_f32(val: i16) -> f32 {
        Self::to_f32_scaled(val, DEFAULT_SCALE)
    }

    pub fn quantize_slice(input: &[f32]) -> Vec<i16> {
        input.iter().map(|&x| Self::to_i16(x)).collect()
    }

    pub fn dequantize_slice(input: &[i16]) -> Vec<f32> {
        input.iter().map(|&x| Self::to_f32(x)).collect()
    }

    /// --- i8 Inference Crunch ---

    #[inline(always)]
    pub fn to_i8(val: f32) -> i8 {
        (val * I8_SCALE).round().clamp(i8::MIN as f32, i8::MAX as f32) as i8
    }

    #[inline(always)]
    pub fn to_f32_i8(val: i8) -> f32 {
        val as f32 / I8_SCALE
    }

    /// Crunches an i16 (Q8.8) into an i8 (Qx.y)
    #[inline(always)]
    pub fn crunch_i16_to_i8(val: i16) -> i8 {
        let f = Self::to_f32(val);
        Self::to_i8(f)
    }

    pub fn quantize_slice_i8(input: &[f32]) -> Vec<i8> {
        input.iter().map(|&x| Self::to_i8(x)).collect()
    }

    pub fn dequantize_slice_i8(input: &[i8]) -> Vec<f32> {
        input.iter().map(|&x| Self::to_f32_i8(x)).collect()
    }
}

/// Block Scaling: 32 elements share 1 scale.
/// This provides much higher dynamic range than global scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block32 {
    pub elements: [i16; 32],
    pub scale: f32,
}

impl Block32 {
    pub fn quantize(data: &[f32]) -> Self {
        let mut max_abs = 0.0f32;
        for &val in data {
            if val.abs() > max_abs { max_abs = val.abs(); }
        }
        
        let scale = if max_abs > 0.0 { 32767.0 / max_abs } else { 1.0 };
        let mut elements = [0i16; 32];
        for i in 0..32 {
            elements[i] = (data[i] * scale).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
        
        Self { elements, scale }
    }

    pub fn dequantize(&self) -> [f32; 32] {
        let mut res = [0.0f32; 32];
        let inv_scale = 1.0 / self.scale;
        for i in 0..32 {
            res[i] = self.elements[i] as f32 * inv_scale;
        }
        res
    }
}


pub struct Calibrator {
    pub min: f32,
    pub max: f32,
}

impl Calibrator {
    pub fn new() -> Self {
        Self { min: f32::MAX, max: f32::MIN }
    }

    pub fn observe(&mut self, val: f32) {
        if val < self.min { self.min = val; }
        if val > self.max { self.max = val; }
    }

    pub fn observe_slice(&mut self, vals: &[f32]) {
        for &v in vals { self.observe(v); }
    }

    pub fn get_scale(&self) -> f32 {
        let range = self.max - self.min;
        if range == 0.0 {
            debug!("Zero range observed during calibration, using default scale");
            return DEFAULT_SCALE;
        }
        let scale = 32767.0 / self.max.abs().max(self.min.abs());
        debug!("Calibrated scale: {}", scale);
        scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization() {
        let val = 1.5;
        let q = Quantizer::to_i16(val);
        assert_eq!(q, (1.5 * 256.0) as i16);
        assert_eq!(Quantizer::to_f32(q), 1.5);
    }

    #[test]
    fn test_i8_crunch() {
        let val = 1.5;
        let q16 = Quantizer::to_i16(val);
        let q8 = Quantizer::crunch_i16_to_i8(q16);
        let val8 = Quantizer::to_f32_i8(q8);
        assert!((val8 - val).abs() < 0.02);
    }

    #[test]
    fn test_calibration() {
        let mut cal = Calibrator::new();
        cal.observe_slice(&[0.0, 1.0, 2.0, 10.0]);
        let scale = cal.get_scale();
        assert!(scale > 0.0);
        let q = Quantizer::to_i16_scaled(10.0, scale);
        assert_eq!(q, 32767);
    }
}
