from torch.nn import Module
from transformers import EncoderDecoderModel, GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutput
from transmodal.generate import generate
from typing import Optional
import os


class GPT2Decoder(Module):
    """
    GPT2 https://huggingface.co/transformers/model_doc/gpt2.html#transformers.GPT2Config.
    """

    is_decoder = True

    def __init__(
        self,
        ckpt_name: str,
        ckpt_dir: Optional[str] = None,
        local_files_only: bool=True,
        **kwargs,
    ):
        """
        Argument/s:
            ckpt_name (str) - name of the checkpoint for the model.
            ckpt_dir (str) - directory containing the pre-trained model checkpoints.
            local_files_only (bool) - initialise from local checkpoints only.
            kwargs - keyword arguments.

        Returns:
            Optionally returns the hidden layer size.
        """
        super(GPT2Decoder, self).__init__()

        # GPT2 configuration
        config = GPT2Config.from_pretrained(
            os.path.join(ckpt_dir, ckpt_name),
            local_files_only=local_files_only,
        )
        config.add_cross_attention = True
        config.is_decoder = True

        # GPT2
        gpt2 = GPT2LMHeadModel.from_pretrained(
            os.path.join(ckpt_dir, ckpt_name),
            config=config,
            local_files_only=local_files_only,
        )

        # Huggingface encoder-to-decoder model
        self.encoder_decoder = EncoderDecoderModel(encoder=gpt2.transformer, decoder=gpt2)
        self.encoder_decoder.decoder.config.hidden_size = self.encoder_decoder.decoder.config.n_embd

        # Remove encoder from encoder-to-decoder model
        del self.encoder_decoder.encoder
        class DummyEncoder:
            class DummyConfig:
                pass
            config = DummyConfig()
            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size
        self.encoder_decoder.encoder = DummyEncoder(hidden_size=self.encoder_decoder.decoder.config.hidden_size)

        # Resize GPT2 embedding to include padding and beginning of sentence token
        self.encoder_decoder.decoder.resize_token_embeddings(config.vocab_size + 2)

    def forward(
        self,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
    ):
        """
        Forward propagation.

        Argument/s:
            encoder_hidden_states - sequence of hidden states from the output of
                the last layer of the encoder.
            encoder_attention_mask - mask to avoid performing cross-attention on
                the padding token indices of the encoder input.
            decoder_input_ids - indices of the tokens for the input sequence.
            decoder_attention_mask - mask to avoid performing attention on
                padding token indices.

        Returns
            outputs - dictionary of outputs.
        """
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # Teacher forcing: labels are given as input
        outputs = self.encoder_decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )
        return {'logits': outputs.logits}

    def generate(
        self,
        max_length,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        num_beams,
        sample,
        log_probs,
        device,
        length_penalty=1.0,
        greedy_search=False,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        """
        Generate predictions autoregresively.

        Argument/s:
            max_length - maximum allowed sequence length.
            bos_token_id - index of the beginning-of-sentence token.
            eos_token_id - index of the end-of-sentence token.
            pad_token_id - index of the padding token.
            num_beams - number of beams for beam search. '1' is a greedy search.
            sample - perform sampling instead of a search.
            log_probs - return the log-probabilities used for sampling (for self-critical sequence training).
            device - which device to place tensors.
            length_penalty - exponential penalty to the length. Values < 1.0 result in shorter sequences, values > 1.0
                result in longer sequences.
            greedy_search (bool) - set the num_beams to one.
            encoder_hidden_states - sequence of hidden states from the output of
                the last layer of the encoder.
            encoder_attention_mask - mask to avoid performing cross-attention on
                the padding token indices of the encoder input.

        Returns:
            Dictionary of outputs.
        """
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        return generate(
            decoder=self.encoder_decoder,
            max_length=max_length,
            length_penalty=length_penalty,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_beams=num_beams,
            sample=sample,
            log_probs=log_probs,
            device=device,
            greedy_search=greedy_search,
            encoder_outputs=encoder_outputs,
        )
