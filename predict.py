import torch
import pickle
import torch.nn as nn
from utils import list2sentence, tensor2sentence, translate_sentence, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from dataset import NumberDataset
from model import Transformer

def predict(sentence):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {DEVICE}")
    max_len = 120
    learning_rate = 3e-4

    zh_vocab_file_path = "zh_vocab.pkl"
    en_vocab_file_path = "en_vocab.pkl"

    with open(zh_vocab_file_path, "rb") as zh_vocab_file:
        zh_vocab = pickle.load(zh_vocab_file)

    with open(en_vocab_file_path, "rb") as en_vocab_file:
        en_vocab = pickle.load(en_vocab_file)
    src_vocab_size = len(zh_vocab)
    trg_vocab_size = len(en_vocab)
    embedding_size = 512
    num_heads = 8
    num_layers = 3
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    forward_expansion = 4
    src_pad_idx = 0
    trg_pad_idx = 0



    zh_ivocab = {index: token for token, index in zh_vocab.items()}
    en_ivocab = {index: token for token, index in en_vocab.items()}
    # load model

    model = Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embedding_size,
        num_layers,
        forward_expansion,
        num_heads,
        dropout,
        DEVICE,
        max_len
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    translated_sentence = translate_sentence(
        model, sentence,  zh_vocab, en_ivocab, DEVICE, max_len=50
    )
    return translated_sentence
    
def main():
    sentence = "首先，也许是最重要的，1989年的革命和随后的苏联解体结束了全球的两极化。"
    translated_sentence = predict(sentence)
    print(translated_sentence)
    
if __name__== "__main__":
    main()