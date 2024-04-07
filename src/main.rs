pub mod config;
pub mod dataset;
pub mod decoder;
pub mod embeddings;
pub mod encoder;
pub mod feed_forward;
pub mod layer_norm;
pub mod multi_head_attention;
pub mod projection_layer;
pub mod residual_connection;
pub mod tokenizer_helper;
pub mod train;
pub mod transformer;
pub mod utils;

use anyhow::Result;
use config::Config;
use train::train_model;

fn main() -> Result<()> {
    let config = Config::default();
    train_model(config)?;

    Ok(())
}
