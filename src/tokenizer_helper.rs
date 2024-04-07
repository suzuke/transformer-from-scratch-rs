use anyhow::{Error, Result};
use tokenizers::models::wordlevel::{WordLevel, WordLevelTrainerBuilder};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{
    tokenizer, AddedToken, DecoderWrapper, NormalizerWrapper, PostProcessorWrapper,
    PreTokenizerWrapper, Tokenizer, TokenizerBuilder,
};

pub struct TokenizerHelper;

impl Default for TokenizerHelper {
    fn default() -> Self {
        Self {}
    }
}

impl TokenizerHelper {
    pub fn get_tokenizer(
        &self,
        lang: &str,
        sequences: Vec<String>,
    ) -> Result<tokenizer::Tokenizer> {
        let filename = format!("tokenizer_{}.json", lang);

        let tokenizer = if std::path::Path::new(&filename).exists() {
            tokenizer::Tokenizer::from_file(&filename).unwrap()
        } else {
            if sequences.is_empty() {
                return Err(Error::msg("sequences is empty"));
            } else {
                self.train(&filename, sequences)?
            }
        };

        Ok(tokenizer)
    }

    fn train(&self, filename: &String, sequences: Vec<String>) -> Result<tokenizer::Tokenizer> {
        let mut trainer = WordLevelTrainerBuilder::default()
            .show_progress(true)
            // .min_frequency(1) // The definition of min_frequency is different with the one in Pytorch
            .special_tokens(vec![
                AddedToken::from("[UNK]", true),
                AddedToken::from("[PAD]", true),
                AddedToken::from("[SOS]", true),
                AddedToken::from("[EOS]", true),
            ])
            .build()
            .unwrap();

        let mut tokenizer = TokenizerBuilder::<
            WordLevel,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        >::default()
        .with_model(
            WordLevel::builder()
                .unk_token("[UNK]".to_string())
                .build()
                .unwrap(),
        )
        .with_pre_tokenizer(Some(PreTokenizerWrapper::Whitespace(Whitespace::default())))
        .build()
        .unwrap();

        // let mut trainer = tokenizer.get_model().get_trainer();

        let pretty = true;
        tokenizer
            .train(&mut trainer, sequences.into_iter())
            .unwrap()
            .save(filename.clone(), pretty)
            .unwrap();

        let tokenizer = Tokenizer::from_file(filename).unwrap();
        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use super::*;

    #[test]
    fn test_tokenizer() {
        let src_lang = "en".to_string();
        let tgt_lang = "pt".to_string();

        let src_sequences = vec![
            "ALICE'S ADVENTURES IN WONDERLAND".to_string(),
            "Lewis Carroll".to_string(),
            "CHAPTER I Down the Rabbit-Hole ,".to_string(),
        ];

        let tgt_sequences = vec![
            "Alice no País das Maravilhas".to_string(),
            "Lewis Carroll".to_string(),
            "Capítulo I Descendo a Toca do Coelho".to_string(),
        ];

        let tokenizer_helper = TokenizerHelper::default();
        let src_tokenizer = tokenizer_helper
            .get_tokenizer(&src_lang, src_sequences.clone())
            .unwrap();

        let tgt_tokenizer = tokenizer_helper
            .get_tokenizer(&tgt_lang, tgt_sequences.clone())
            .unwrap();

        zip(src_sequences, tgt_sequences).for_each(|(src, tgt)| {
            let src_tokens = src_tokenizer.encode(src.clone(), true).unwrap();
            let src_ids = src_tokens.get_ids();
            let tgt_tokens = tgt_tokenizer.encode(tgt.clone(), true).unwrap();
            let tgt_ids = tgt_tokens.get_ids();
            // println!("src_tokens: {:?}", src_tokens);
            // println!("src_tokens: {:?}", src_ids);
            // println!("tgt_tokens: {:?}", tgt_tokens);
            // println!("tgt_tokens: {:?}", tgt_ids);

            let src_text = src_tokenizer.decode(src_ids, false).unwrap();
            let tgt_text = tgt_tokenizer.decode(tgt_ids, false).unwrap();

            println!("src {:?}, src_text: {:?}", src, src_text);
            println!("tgt {:?}, tgt_text: {:?}", tgt, tgt_text);

            // assert_eq!(src_text.len(), src.len());
            // assert_eq!(tgt_text.len(), tgt.len());
        });
    }
}
