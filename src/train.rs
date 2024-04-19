use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
use candle_nn::{loss, AdamW, Optimizer, VarBuilder, VarMap};
use tokenizers::tokenizer::Tokenizer;

use crate::{
    config::Config,
    dataset::{BilingualDataset, OpusBooksDataset},
    tokenizer_helper::TokenizerHelper,
    transformer::{transformer, Transformer},
    utils::{causal_mask, device},
};

pub fn greedy_decode(
    model: &Transformer,
    src: &Tensor,
    src_mask: &Tensor,
    src_tokenizer: &Tokenizer,
    tgt_tokenizer: &Tokenizer,
    max_tokens: usize,
    device: &Device,
) -> Result<Tensor> {
    let sos_id = tgt_tokenizer.token_to_id("[SOS]").unwrap();
    let eos_id = tgt_tokenizer.token_to_id("[EOS]").unwrap();

    let encoder_output = model.encode(src, src_mask, false)?; //(batch, seq_len, d_model)
    println!("encoder_output: {:?}", encoder_output.shape());
    let mut decoder_input = Tensor::new(sos_id, device)?
        .reshape((1, 1))? //(batch=1, seq_len=1) -> [[sos]]
        .to_dtype(src.dtype())?;

    while decoder_input.dims()[1] < max_tokens {
        println!("decoder_input: {:?}", decoder_input.shape());

        let decoder_mask = causal_mask(decoder_input.dims()[1], device)?;
        println!("decoder_mask: {:?}", decoder_mask.shape());

        let decoder_output = model.decode(
            &encoder_output,
            src_mask,
            &decoder_input,
            &decoder_mask,
            false,
        )?; //(batch, seq_len, d_model)
        println!("decoder_output: {:?}", decoder_output.shape());

        let decoder_output = decoder_output
            .i((
                ..,
                decoder_output.dims()[1] - 1..,
                ..,
            ))
            .unwrap()
            .squeeze(0)
            .unwrap();
        println!("decoder_output[-1]: {:?}", decoder_output.shape());
        let prob = model.project(&decoder_output)?;
        println!("prob: {:?}", prob.shape());
        println!("prob argmax: {:?}", prob.argmax(D::Minus1)?.to_vec1::<u32>());

        let next_word = prob
            .argmax(D::Minus1)?
            .squeeze(0)?
            .to_vec0::<u32>()?;
        // println!("next_word: {:?}", next_word);

        decoder_input = Tensor::cat(
            &[
                &decoder_input,
                &Tensor::new(next_word, device)?
                    .to_dtype(src.dtype())?
                    .reshape((1, 1))?,
            ],
            1,
        )?;

        if next_word == eos_id {
            break;
        }
    }
    Ok(decoder_input) //(batch=1, seq_len=max_tokens)
}

pub fn train_model(config: Config) -> Result<()> {
    let device = device(false)?;
    let dtype = DType::F32;

    let dataset = OpusBooksDataset::new(config.src_lang.clone(), config.tgt_lang.clone(), 0.9)?;

    println!("train dataset: {:?}", dataset.train_set().len());
    println!("valid dataset: {:?}", dataset.valid_set().len());

    let tokenizer_helper = TokenizerHelper::default();
    let tokenizer_src =
        tokenizer_helper.get_tokenizer(&config.src_lang, dataset.sequences(&config.src_lang))?;
    let src_vocab_size = tokenizer_src.get_vocab_size(true);
    println!("src_vocab size: {}", src_vocab_size);

    let tokenizer_tgt =
        tokenizer_helper.get_tokenizer(&config.tgt_lang, dataset.sequences(&config.tgt_lang))?;
    let tgt_vocab_size = tokenizer_tgt.get_vocab_size(true);
    println!("tgt_vocab size: {}", tgt_vocab_size);

    let dataset = BilingualDataset::new(
        &dataset,
        &tokenizer_src,
        &tokenizer_tgt,
        config.seq_len,
        &device,
    )?;
    println!("dataset: {:?}", dataset.train_set.len());
    println!("dataset: {:?}", dataset.valid_set.len());

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let model = transformer(
        src_vocab_size,
        tgt_vocab_size,
        config.seq_len,
        config.seq_len,
        config.d_model,
        config.n,
        config.num_heads,
        config.drop_p,
        config.d_ff,
        vb,
    )?;

    let model_path = format!("./model/{0}_{1}_checkpoint.safetensors", config.src_lang, config.tgt_lang);
    if Path::new(&model_path).exists() {
        println!("loading model from: {}", model_path);
        varmap.load(&model_path)?;
    }

    let mut optimizer = AdamW::new_lr(varmap.all_vars(), config.lr)?;

    for epoch in 0..config.num_epochs {
        let train_set_batcher = dataset.train_batcher(config.batch_size);
        println!("epoch: {}", epoch);
        for (encoder_input, decoder_input, encoder_mask, decoder_mask, label) in train_set_batcher {
            let encoder_output = model.encode(&encoder_input, &encoder_mask, false)?; //(batch, seq_len, d_model)

            let decoder_output = model.decode(
                &encoder_output,
                &encoder_mask,
                &decoder_input,
                &decoder_mask,
                false,
            )?; //(batch, seq_len, d_model)
            let proj_output = model.project(&decoder_output)?; //(batch, seq_len, vocab_size)
            let loss = loss::cross_entropy(
                &proj_output.flatten_to(1)?,
                &label.flatten_to(1)?,
            )?;
            let _ = optimizer.backward_step(&loss);

            println!("loss: {:?}", loss.to_vec0::<f32>());
        }

        // validation
        if epoch % 1 == 0 {
            let valid_set_batcher = dataset.valid_batcher(1);
            for (encoder_input, _, encoder_mask, _, label) in
                valid_set_batcher.take(3)
            {
                let model_out = greedy_decode(
                    &model,
                    &encoder_input,
                    &encoder_mask,
                    &tokenizer_src,
                    &tokenizer_tgt,
                    10, //config.seq_len,
                    &device,
                )?;
                println!("model_out: {:?}", model_out.squeeze(0)?.to_vec1::<u32>()?);

                let skip_special_tokens = false;
                let model_out_text =
                    ids_to_text(&tokenizer_tgt, &model_out.squeeze(0)?, skip_special_tokens);
                let encoder_input_text = ids_to_text(
                    &tokenizer_src,
                    &encoder_input.squeeze(0)?,
                    skip_special_tokens,
                );
                let label_text =
                    ids_to_text(&tokenizer_tgt, &label.squeeze(0)?, skip_special_tokens);

                println!("encoder_input_text: {:?}", encoder_input_text);
                println!("label_text: {:?}", label_text);
                println!("model_out_text: {:?}", model_out_text);
            }
        }

        // save model
        varmap.save(&model_path)?
    }

    Ok(())
}

fn ids_to_text(tokenizer: &Tokenizer, ids: &Tensor, skip_special_tokens: bool) -> String {
    tokenizer
        .decode(
            ids.to_vec1::<u32>().unwrap().as_slice(),
            skip_special_tokens,
        )
        .unwrap()
}
