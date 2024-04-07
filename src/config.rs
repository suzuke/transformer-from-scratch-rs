#[derive(Debug)]
pub struct Config {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub n: usize,
    pub num_heads: usize,
    pub drop_p: f32,
    pub d_ff: usize,
    pub src_lang: String,
    pub tgt_lang: String,
    pub lr: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_epochs: 20,
            batch_size: 64,
            seq_len: 350,
            d_model: 128,
            n: 4,
            num_heads: 4,
            drop_p: 0.1,
            d_ff: 2048,
            src_lang: "en".to_string(),
            tgt_lang: "it".to_string(),
            lr: 0.0001f64,
        }
    }
}
