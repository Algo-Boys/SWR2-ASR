"""Decoder for CTC-based ASR.""" ""
import torch

from swr2_asr.utils.tokenizer import CharTokenizer
from swr2_asr.utils.data import create_lexicon
import os
from torchaudio.datasets.utils import _extract_tar
from torchaudio.models.decoder import ctc_decoder
LEXICON = "lexicon.txt"
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

def beam_search_decoder(output, tokenizer:CharTokenizer, tokenizer_txt_path,lang_model_path):
    if not os.path.isdir(lang_model_path):
        url = f"https://dl.fbaipublicfiles.com/mls/mls_lm_german.tar.gz"
        torch.hub.download_url_to_file(
                url, "data/mls_lm_german.tar.gz" )
        _extract_tar("data/mls_lm_german.tar.gz", overwrite=True)
    if  not os.path.isfile(tokenizer_txt_path):
        tokenizer.create_txt(tokenizer_txt_path)
    
    lexicon_path= os.join(lang_model_path, LEXICON)
    if not os.path.isfile(lexicon_path):
        occurences_path = os.join(lang_model_path,"vocab_counts.txt")
        create_lexicon(occurences_path, lexicon_path)
    lm_path = os.join(lang_model_path,"3-gram_lm.apa")
    decoder = ctc_decoder(lexicon = lexicon_path,
                           tokenizer = tokenizer_txt_path,
                           lm =lm_path,
                           blank_token = '_', 
                           nbest =1,
                           sil_token= '<SPACE>', 
                           unk_word = '<UNK>')
    return decoder