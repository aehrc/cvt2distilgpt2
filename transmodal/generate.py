from torch.distributions.categorical import Categorical
from transformers.modeling_outputs import BaseModelOutput
import torch


def generate(
        decoder,
        max_length,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        num_beams,
        sample,
        log_probs,
        device,
        length_penalty=1.0,
        output_attentions=False,
        greedy_search=False,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_outputs=None,
):
    """
    Generate predictions autoregresively.

    Argument/s:
        decoder - Hugging Face decoder.
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
        output_attentions (bool) - Return the attentions tensors of all attention layers.
        greedy_search (bool) - sets num_beams to one.
        encoder_hidden_states - sequence of hidden states from the output of
            the last layer of the encoder.
        encoder_attention_mask - mask to avoid performing cross-attention on
            the padding token indices of the encoder input.
        encoder_outputs (ModelOutput) - encoder_hidden_stats and encoder_attention mask wrapped in a huggingface
            ModelOutput class.

    Returns:
        Dictionary of outputs.
    """

    # Wrap encoder variables in BaseModelOutput
    if encoder_hidden_states:
        assert encoder_outputs is None, 'Cannot set ''encoder_hidden_states'' and ''encoder_outputs'''
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

    # Greedy search (beam search with one beam)
    if greedy_search:
        num_beams = 1

    # Beam search is not used with sample_log_probs()
    if sample and log_probs:
        assert num_beams == 1, 'Beam search is not used when sample and log_probs are set.'

    # Replication required for beam search
    if num_beams > 1 and encoder_attention_mask is not None:
        expanded_return_idx = (
            torch.arange(encoder_hidden_states.size()[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(device)
        )
        encoder_attention_mask = encoder_attention_mask.index_select(0, expanded_return_idx)

    # Return log-probabilities for self-critical sequence training.
    if sample and log_probs:
        bos_ids = (
                torch.ones(
                    (encoder_outputs.last_hidden_state.size()[0], 1),
                    dtype=torch.long,
                    device=device,
                )
                * bos_token_id
        )
        return sample_log_probs(
            decoder=decoder,
            input_ids=bos_ids,
            max_length=max_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

    # Autoregresively generate predictions
    return decoder.generate(
        max_length=max_length,
        length_penalty=length_penalty,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        num_beams=num_beams,
        return_dict_in_generate=True,
        do_sample=sample,
        use_cache=True,
        output_attentions=output_attentions,
        encoder_outputs=encoder_outputs,
    )

def sample_log_probs(
    decoder,
    input_ids,
    max_length,
    eos_token_id,
    pad_token_id,
    encoder_hidden_states,
    encoder_attention_mask,
):
    '''
    Generates sequences using multinomial sampling. Returns the
    log-probabilities used for sampling.

    Argument/s:
        decoder - Hugging Face decoder.
        input_ids - indices of the tokens for the input sequence.
        max_length - maximum allowed sequence length.
        eos_token_id - index of the end-of-sentence token.
        pad_token_id - index of the padding token.
        encoder_hidden_states - sequence of hidden states from the output of
            the last layer of the encoder.
        encoder_attention_mask - mask to avoid performing cross-attention on
            the padding token indices of the encoder input.

    Returns:
        Dictionary containing the sample sequences and the log-probabilities.
    '''
    log_probs = []
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    sequence_lengths = input_ids.new(input_ids.shape[0]).fill_(max_length)
    cur_len = input_ids.shape[-1]
    past_key_values = None

    while cur_len < max_length:

        outputs = decoder(
            input_ids=input_ids[:, -1:],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        # Sample the next token.
        pmf = Categorical(logits=next_token_logits)
        next_token_ids = pmf.sample()
        log_prob = pmf.log_prob(next_token_ids)
        log_probs.append(log_prob)

        next_token_ids = next_token_ids * unfinished_sequences + (pad_token_id) * (
            1 - unfinished_sequences
        )

        input_ids = torch.cat([input_ids, next_token_ids.unsqueeze(1)], dim=-1)

        cur_len = cur_len + 1
        is_eos_in_next_token = next_token_ids == eos_token_id
        is_sent_unfinished = unfinished_sequences.mul(
            is_eos_in_next_token.long()
        ).bool()
        sequence_lengths = sequence_lengths.masked_fill(is_sent_unfinished, cur_len)
        unfinished_sequences = unfinished_sequences.mul(
            (~is_eos_in_next_token).long()
        )

        # Stop when EOS is in each sentence, or if max_length is exceeded
        if unfinished_sequences.max() == 0:
            break

    return {'samples': input_ids, 'log_probs': torch.stack(log_probs, dim=-1)}