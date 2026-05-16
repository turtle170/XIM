use crate::quant::DEFAULT_SCALE;

pub struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    pub fn new(seed: u32) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    #[inline(always)]
    pub fn next(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Returns a float in [0.0, 1.0)
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        let int_val = self.next() & 0xFFFFFF;
        (int_val as f32) / (0xFFFFFF as f32)
    }
}

pub struct StochasticQuantizer {
    rng: Xorshift32,
}

impl StochasticQuantizer {
    pub fn new(seed: u32) -> Self {
        Self { rng: Xorshift32::new(seed) }
    }

    #[inline(always)]
    pub fn to_i16_stochastic(&mut self, val: f32, scale: f32) -> i16 {
        let scaled = val * scale;
        let floor = scaled.floor();
        let prob = scaled - floor;
        
        let round_up = if self.rng.next_f32() < prob { 1.0 } else { 0.0 };
        
        (floor + round_up).clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }
}
