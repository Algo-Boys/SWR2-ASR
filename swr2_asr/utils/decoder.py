"""Decoder for CTC-based ASR.""" ""
import torch

from swr2_asr.utils.tokenizer import CharTokenizer


# TODO: refactor to use torch CTC decoder class
def greedy_decoder(output, labels, label_lengths, tokenizer: CharTokenizer, collapse_repeated=True):
    """Greedily decode a sequence."""
    blank_label = tokenizer.get_blank_token()
    arg_maxes = torch.argmax(output, dim=2)  # pylint: disable=no-member
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(tokenizer.decode(labels[i][: label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(tokenizer.decode(decode))
    return decodes, targets


# TODO: add beam search decoder

