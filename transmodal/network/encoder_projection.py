from torch.nn import Linear, Module
from typing import Dict, List, Union
import torch


class EncoderPermuteProject(Module):
    """
    Permutes the last hidden state of the encoder so that the spatial position is represented by axis -2 and the encoded
    representation for a spatial position is represented by axis -1. Following this, the encoded representation for each
    spatial position is projected to the size of the decoder's hidden state.
    """
    def __init__(
        self,
        permute_encoder_last_hidden_state: Union[List, bool],
        encoder_last_hidden_state_size: int,
        decoder_hidden_state_size: int,
        **kwargs,
    ):
        """
        Argument/s:
            permute_encoder_last_hidden_state - permutation of the last hidden state of the encoder.
            encoder_last_hidden_state_size - the size of the encoder's last hidden state for each spatial position,
                or axis -1.
            decoder_hidden_state_size - the hidden state size of the decoder.
            kwargs - keyword arguments.
        """
        super(EncoderPermuteProject, self).__init__()

        self.permute_encoder_last_hidden_state = permute_encoder_last_hidden_state
        self.projection = Linear(
            in_features=encoder_last_hidden_state_size,
            out_features=decoder_hidden_state_size,
            bias=False,
        )

    def forward(self, encoder_last_hidden_state: torch.FloatTensor) -> Dict[str, torch.FloatTensor]:
        """
        Forward propagation.

        Argument/s:
            images - a batch of images.

        Returns
            Dictionary of outputs.
        """
        if self.permute_encoder_last_hidden_state:
            encoder_last_hidden_state = encoder_last_hidden_state.permute(self.permute_encoder_last_hidden_state)
        return {"projected_encoder_last_hidden_state": self.projection(encoder_last_hidden_state)}
