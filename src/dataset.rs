use candle_core::{Device, Error, Result, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use parquet::file::reader::SerializedFileReader;
use std::{collections::HashMap, fs::File};
use tokenizers::Tokenizer;

use crate::utils::{causal_mask, device};

// https://huggingface.co/datasets/Helsinki-NLP/opus_books
#[derive(Debug)]
pub struct OpusBooksDataset {
    pub src_lang: String,
    pub tgt_lang: String,
    pub train_set: Vec<HashMap<String, String>>, // vec![(src_lang: String, tgt_lang: String)]>
    pub valid_set: Vec<HashMap<String, String>>,
}

impl OpusBooksDataset {
    pub fn new(src_lang: String, tgt_lang: String, train_split: f32) -> Result<Self> {
        let buffer = OpusBooksDataset::load(&src_lang, &tgt_lang)?;
        let train_size = (buffer.len() as f32 * train_split).round() as usize;
        let (train_set, valid_set) = buffer.split_at(train_size);

        Ok(Self {
            src_lang,
            tgt_lang,
            train_set: train_set.to_vec(),
            valid_set: valid_set.to_vec(),
        })
    }

    pub fn train_set(&self) -> &Vec<HashMap<String, String>> {
        &self.train_set
    }

    pub fn valid_set(&self) -> &Vec<HashMap<String, String>> {
        &self.valid_set
    }

    pub fn sequences(&self, lang: &String) -> Vec<String> {
        self.train_set
            .iter()
            .filter_map(|map| map.get(lang).cloned())
            .collect()
    }

    fn load(src_lang: &String, tgt_lang: &String) -> Result<Vec<HashMap<String, String>>> {
        let dataset_id = "Helsinki-NLP/opus_books".to_string();
        let api = Api::new().map_err(|e| Error::Msg(format!("Api error: {e}")))?;
        let repo = Repo::with_revision(
            dataset_id,
            RepoType::Dataset,
            "refs/convert/parquet".to_string(),
        );
        let repo = api.repo(repo);
        let local = repo
            .get(&format!("{}-{}/train/0000.parquet", src_lang, tgt_lang))
            .map_err(|e| Error::Msg(format!("Api error: {e}")))?;
        let file = File::open(local)?;
        let parquet_reader = SerializedFileReader::new(file)
            .map_err(|e| Error::Msg(format!("Parquet error: {e}")))?;
        Ok(Self::load_parquet(parquet_reader))
    }

    fn load_parquet(parquet: SerializedFileReader<File>) -> Vec<HashMap<String, String>> {
        parquet
            .into_iter()
            .flatten()
            .flat_map(|row| {
                row.get_column_iter()
                    .filter_map(|(_name, field)| {
                        if let parquet::record::Field::Group(subrow) = field {
                            let mut map = HashMap::new();
                            for (lang, translation) in subrow.get_column_iter() {
                                if let parquet::record::Field::Str(translation) = translation {
                                    map.insert(lang.clone(), translation.clone());
                                }
                            }
                            Some(map)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BilingualItem {
    encoder_input: Tensor,
    decoder_input: Tensor,
    encoder_mask: Tensor,
    decoder_mask: Tensor,
    label: Tensor,
    src_text: String,
    tgt_text: String,
}

fn raw_to_bilingual(
    raw_set: &Vec<HashMap<String, String>>,
    tokenizer_src: &Tokenizer,
    tokenizer_tgt: &Tokenizer,
    src_lang: &String,
    tgt_lang: &String,
    sos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    seq_len: usize,
    device: &Device,
) -> Result<Vec<BilingualItem>> {
    let bilingual_items = raw_set
        .iter()
        .map(|map| {
            let src_text = map.get(src_lang).unwrap();
            let tgt_text = map.get(tgt_lang).unwrap();

            let enc_input_encoding = tokenizer_src.encode(src_text.clone(), true).unwrap();
            let enc_input_tokens = enc_input_encoding.get_ids();
            let dec_input_encoding = tokenizer_tgt.encode(tgt_text.clone(), true).unwrap();
            let dec_input_tokens = dec_input_encoding.get_ids();

            // Add sos, eos and padding to earch sequence
            let enc_num_padding_tokens = seq_len - enc_input_tokens.len() - 2; // -2 for <s> and </s>
                                                                               // Only add <s>, and </s> only on the label
            let dec_num_padding_tokens = seq_len - dec_input_tokens.len() - 1; // -1 for <s> or </s>

            assert!(
                enc_num_padding_tokens as i32 >= 0 && dec_num_padding_tokens as i32 >= 0,
                "Sentence is too long!"
            );

            let eos_token = Tensor::new(&[eos_token_id], &device).unwrap();
            let sos_token = Tensor::new(&[sos_token_id], &device).unwrap();
            let encoder_padding = Tensor::from_vec(
                vec![pad_token_id; enc_num_padding_tokens],
                (enc_num_padding_tokens,),
                &device,
            )
            .unwrap();
            let decoder_padding = Tensor::from_vec(
                vec![pad_token_id; dec_num_padding_tokens],
                (dec_num_padding_tokens,),
                &device,
            )
            .unwrap();
            let decoder_input_tokens = Tensor::new(dec_input_tokens, &device).unwrap();

            // Add [SOS] and [EOS] token and padding to the input
            let encoder_input = Tensor::cat(
                &[
                    &sos_token,
                    &Tensor::new(enc_input_tokens, &device).unwrap(),
                    &eos_token,
                    &encoder_padding,
                ],
                0,
            )
            .unwrap();

            // Add only [SOS] token and padding to the label
            let decoder_input =
                Tensor::cat(&[&sos_token, &decoder_input_tokens, &decoder_padding], 0).unwrap();

            // Add only [EOS] token and padding to the label
            let label =
                Tensor::cat(&[&decoder_input_tokens, &eos_token, &decoder_padding], 0).unwrap();

            // make sure all tensors have the same shape: (seq_len)
            assert_eq!(encoder_input.dims()[0], seq_len);
            assert_eq!(decoder_input.dims()[0], seq_len);
            assert_eq!(label.dims()[0], seq_len);

            //verify correctness and refactor needed!!
            let decoder_input_ne = decoder_input
                .ne(pad_token_id) //
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let casual_mask = causal_mask(seq_len, &device).unwrap();
            let decoder_mask = decoder_input_ne.broadcast_mul(&casual_mask).unwrap();
            // println!("decoder_mask: {:?}", decoder_mask.to_vec3::<u8>());

            let encoder_mask = encoder_input
                .ne(pad_token_id)
                .unwrap()
                .reshape((1, 1, seq_len))
                .unwrap();

            BilingualItem {
                encoder_input, //(seq_len)
                decoder_input, //(seq_len)
                encoder_mask,  //(1, 1, seq_len)
                decoder_mask,  //(1, 1, seq_len, seq_len)
                label,         //(seq_len)
                src_text: src_text.clone(),
                tgt_text: tgt_text.clone(),
            }
        })
        .collect();
    Ok(bilingual_items)
}

#[derive(Debug, Clone)]
pub struct BilingualDataset {
    pub src_lang: String,
    pub tgt_lang: String,
    pub train_set: Vec<BilingualItem>,
    pub valid_set: Vec<BilingualItem>,
}

impl BilingualDataset {
    pub fn new(
        raw_dataset: &OpusBooksDataset,
        tokenizer_src: &Tokenizer,
        tokenizer_tgt: &Tokenizer,
        seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let src_lang = raw_dataset.src_lang.clone();
        let tgt_lang = raw_dataset.tgt_lang.clone();

        let sos_token_id = tokenizer_tgt.token_to_id("[SOS]").expect("[SOS] not found");
        let eos_token_id = tokenizer_tgt.token_to_id("[EOS]").expect("[EOS] not found");
        let pad_token_id = tokenizer_tgt.token_to_id("[PAD]").expect("[PAD] not found");

        let train_set = raw_to_bilingual(
            &raw_dataset.train_set,
            tokenizer_src,
            tokenizer_tgt,
            &src_lang,
            &tgt_lang,
            sos_token_id,
            eos_token_id,
            pad_token_id,
            seq_len,
            device,
        )?;
        let valid_set = raw_to_bilingual(
            &raw_dataset.valid_set,
            tokenizer_src,
            tokenizer_tgt,
            &src_lang,
            &tgt_lang,
            sos_token_id,
            eos_token_id,
            pad_token_id,
            seq_len,
            device,
        )?;

        Ok(Self {
            src_lang,
            tgt_lang,
            train_set,
            valid_set,
        })
    }

    pub fn train_batcher(&self, batch_size: usize) -> BilingualBatcher {
        BilingualBatcher::new(&self.train_set, batch_size)
    }

    pub fn valid_batcher(&self, batch_size: usize) -> BilingualBatcher {
        BilingualBatcher::new(&self.valid_set, batch_size)
    }
}

pub struct BilingualBatcher<'a> {
    // dataset: &'a [BilingualItem],
    dataset: Vec<&'a BilingualItem>,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> BilingualBatcher<'a> {
    pub fn new(dataset: &'a [BilingualItem], batch_size: usize) -> Self {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut dataset = dataset.iter().collect::<Vec<_>>();
        dataset.shuffle(&mut thread_rng());

        Self {
            dataset,
            batch_size,
            current_idx: 0,
        }
    }
}

impl Iterator for BilingualBatcher<'_> {
    type Item = (Tensor, Tensor, Tensor, Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        // Calculate end index for the next batch. This could be smaller than `batch_size` if we're at the end of the dataset.
        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let batch_items = &self.dataset[self.current_idx..end_idx];

        let batch_encoder_input = Tensor::stack(
            &batch_items
                .iter()
                .map(|item| item.encoder_input.clone())
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();
        let batch_decoder_input = Tensor::stack(
            &batch_items
                .iter()
                .map(|item| item.decoder_input.clone())
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();
        let batch_encoder_mask = Tensor::stack(
            &batch_items
                .iter()
                .map(|item| item.encoder_mask.clone())
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();
        let batch_decoder_mask = Tensor::stack(
            &batch_items
                .iter()
                .map(|item| item.decoder_mask.clone())
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();
        let batch_label = Tensor::stack(
            &batch_items
                .iter()
                .map(|item| item.label.clone())
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();

        // Update the current index for the next call to `next`.
        self.current_idx = end_idx;

        Some((
            batch_encoder_input,
            batch_decoder_input,
            batch_encoder_mask,
            batch_decoder_mask,
            batch_label,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp::max, iter::zip};

    use crate::tokenizer_helper::TokenizerHelper;

    use super::*;

    #[test]
    fn test_bilingual_batcher() {
        let src_lang = "en".to_string();
        let tgt_lang = "pt".to_string();
        let device = Device::Cpu;
        let raw_dataset =
            OpusBooksDataset::new(src_lang.clone(), tgt_lang.clone(), 0.9f32).unwrap();
        let max_sequences = 3;
        let src_sequences = raw_dataset.sequences(&src_lang);
        let src_sequences = src_sequences.iter().take(max_sequences);
        let tgt_sequences = raw_dataset.sequences(&tgt_lang);
        let tgt_sequences = tgt_sequences.iter().take(max_sequences);
        let tokenizer_helper = TokenizerHelper::default();
        let src_tokenizer = tokenizer_helper
            .get_tokenizer(&src_lang, raw_dataset.sequences(&src_lang))
            .unwrap();
        let tgt_tokenizer = tokenizer_helper
            .get_tokenizer(&tgt_lang, raw_dataset.sequences(&tgt_lang))
            .unwrap();

        // calculate max sequence length
        let max_src_len = src_sequences
            .clone()
            .map(|s| {
                src_tokenizer
                    .encode(s.clone(), true)
                    .unwrap()
                    .get_ids()
                    .len()
            })
            .max()
            .unwrap();

        let max_tgt_len = tgt_sequences
            .clone()
            .map(|s| {
                tgt_tokenizer
                    .encode(s.clone(), true)
                    .unwrap()
                    .get_ids()
                    .len()
            })
            .max()
            .unwrap();
        // let seq_len = 50;
        let seq_len = max(max_src_len, max_tgt_len) + 4; // 2 for <s>, </s> and padding
        println!("seq_len: {}", seq_len);

        let raw_train_set = zip(src_sequences, tgt_sequences)
            .map(|(src, tgt)| {
                let mut map = HashMap::new();
                // println!("src: {}, tgt: {}", src, tgt);
                map.insert(src_lang.clone(), src.clone());
                map.insert(tgt_lang.clone(), tgt.clone());
                map
            })
            .collect::<Vec<HashMap<String, String>>>();

        let sos_token_id = tgt_tokenizer.token_to_id("[SOS]").expect("[SOS] not found");
        let eos_token_id = tgt_tokenizer.token_to_id("[EOS]").expect("[EOS] not found");
        let pad_token_id = tgt_tokenizer.token_to_id("[PAD]").expect("[PAD] not found");

        let train_set = raw_to_bilingual(
            &raw_train_set,
            &src_tokenizer,
            &tgt_tokenizer,
            &src_lang,
            &tgt_lang,
            sos_token_id,
            eos_token_id,
            pad_token_id,
            seq_len,
            &device,
        )
        .unwrap();

        let dataset = BilingualDataset {
            src_lang,
            tgt_lang,
            train_set,
            valid_set: vec![],
        };

        // test batcher for different batch sizes
        (1..=3).step_by(2).for_each(|batch_size| {
            1;
            let batcher = dataset.train_batcher(batch_size);
            for (encoder_input, decoder_input, encoder_mask, decoder_mask, label) in batcher {
                assert_eq!(encoder_input.dims(), &[batch_size, seq_len]);
                assert_eq!(decoder_input.dims(), &[batch_size, seq_len]);
                assert_eq!(encoder_mask.dims(), &[batch_size, 1, 1, seq_len]);
                assert_eq!(decoder_mask.dims(), &[batch_size, 1, seq_len, seq_len]);
                assert_eq!(label.dims(), &[batch_size, seq_len]);

                // println!("encoder_input: {:?}", encoder_input.to_vec2::<u32>().unwrap());
                // println!("encoder_mask: {:?}", encoder_mask.squeeze(0).unwrap().to_vec3::<u8>().unwrap());

                // let encoder_input_text = src_tokenizer
                //     .decode(
                //         encoder_input
                //             .squeeze(0)
                //             .unwrap()
                //             .to_vec1::<u32>()
                //             .unwrap()
                //             .as_slice(),
                //         false,
                //     )
                //     .unwrap();
                // let label_text = tgt_tokenizer
                //     .decode(
                //         label
                //             .squeeze(0)
                //             .unwrap()
                //             .to_vec1::<u32>()
                //             .unwrap()
                //             .as_slice(),
                //         false,
                //     )
                //     .unwrap();

                // println!("encoder_input_text: {:?}", encoder_input_text);
                // println!("label_text: {:?}", label_text);
            }
        });
    }
}

// encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
// decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
// encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
// decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
// label = batch['label'].to(device) # (B, seq_len)
// src: ALICE'S ADVENTURES IN WONDERLAND, tgt: Alice no País das Maravilhas
// src: Lewis Carroll, tgt: Lewis Carroll
// src: CHAPTER I Down the Rabbit-Hole, tgt: Capítulo I Descendo a Toca do Coelho
