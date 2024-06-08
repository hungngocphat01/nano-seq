import torch
from nano_seq.data import Dictionary
from nano_seq.data.utils import get_decoder_mask
from nano_seq.model.translation import TranslationModel


def iterative_greedy_single(
    model: TranslationModel, h_encoder: torch.Tensor, enc_mask: torch.Tensor, tgt_dict: Dictionary, max_length=100
):
    decoder_input = torch.Tensor([[tgt_dict.sos]]).long().cuda()

    for _ in range(max_length):
        # Decoder prediction
        h_decoder = model.decoder(decoder_input, h_encoder, get_decoder_mask(decoder_input), enc_mask)
        logits = model.out_project(h_decoder)

        # Greedy selection
        token_index = torch.argmax(logits[:, -1], keepdim=True)

        # EOS is most probable => Exit
        if token_index.item() == tgt_dict.eos:
            break

        # Next Input to Decoder
        decoder_input = torch.cat([decoder_input, token_index], dim=1)

    decoder_output_final = decoder_input[0, 1:].cpu().numpy()
    out_sent = " ".join(tgt_dict.decode(decoder_output_final))

    return out_sent


def batched_greedy(
    model: TranslationModel, h_encoder: torch.Tensor, enc_mask: torch.Tensor, tgt_dict: Dictionary, max_length=512
) -> torch.Tensor:
    """
    Perform batched greedy decoding

    Args
    ----
    model: TranslationModel

    h_encoder: tensor
        hidden state of encoder, of shape [bsz, max_len, d_model]
    enc_mask: tensor
        self-attention mask of source sentences, of shape [bsz, 1, 1, max_len]
    tgt_dict: Dictionary
        dictionary of target language
    max_length: int, default = 512
        maximum length allowed for an output sentence, to prevent OOM if the decoder
        just bla bla bla bla for too long

    Returns
    -------
    tensor
        of shape [bsz, max_len]
        each element is the predicted token idx
    """
    decoder_input = torch.Tensor([[tgt_dict.sos]] * len(h_encoder)).long().cuda()
    completed_mask = torch.zeros(len(decoder_input), dtype=torch.bool)

    for _ in range(max_length):
        with torch.no_grad():
            h_decoder = model.decoder(decoder_input, h_encoder, get_decoder_mask(decoder_input), enc_mask)
            logits = model.out_project(h_decoder)

        # most probable token at this timestep
        token_index = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).long()
        decoder_input = torch.cat([decoder_input, token_index], dim=1)

        # which sentences returned eos now?
        step_eos_mask = (token_index == tgt_dict.eos).squeeze(-1)

        # update which sentences are already completed
        completed_mask = torch.logical_or(completed_mask, step_eos_mask)

        # all sentences decoded without exceeding max_length
        if torch.all(completed_mask).item():
            return decoder_input

    # any sentence exceeding max_length should end with eos
    decoder_input = torch.cat([decoder_input, torch.zeros(len(decoder_input), 1).long() + tgt_dict.eos], dim=1)
    return decoder_input
