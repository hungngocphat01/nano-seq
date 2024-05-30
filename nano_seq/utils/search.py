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
