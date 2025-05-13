import torch
import pickle
import torch.nn as nn
from utils import list2sentence, tensor2sentence, translate_sentence, save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from dataset import NumberDataset
from model import Transformer
import spacy
#from sklearn.model_selection import train_test_split


# hyperparameters 1

BATCH_SIZE = 32
EPOCHS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 3e-4
max_len = 120


# get vocabulary

def test1():
    zh_vocab_file_path = "zh_vocab.pkl"
    en_vocab_file_path = "en_vocab.pkl"

    with open(zh_vocab_file_path, "rb") as zh_vocab_file:
        zh_vocab = pickle.load(zh_vocab_file)

    with open(en_vocab_file_path, "rb") as en_vocab_file:
        en_vocab = pickle.load(en_vocab_file)

    zh_ivocab = {index: token for token, index in zh_vocab.items()}
    en_ivocab = {index: token for token, index in en_vocab.items()}

    src_file = 'data/chinese.zh/chinese.zh'
    trg_file = 'data/english.en/english.en'


    # get dataset

    dataset = NumberDataset(src_file, trg_file, zh_vocab, en_vocab, max_len)

    '''
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    '''

    train_loader =  DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    for idx, (src, trg) in enumerate(train_loader):

        print(src[0])
        print(trg[0])
        if idx == 10:
            break
        
def test2():
    zh_vocab_file_path = "zh_vocab.pkl"
    en_vocab_file_path = "en_vocab.pkl"

    with open(zh_vocab_file_path, "rb") as zh_vocab_file:
        zh_vocab = pickle.load(zh_vocab_file)

    with open(en_vocab_file_path, "rb") as en_vocab_file:
        en_vocab = pickle.load(en_vocab_file)


    src_file = 'data/chinese.zh/chinese.zh'
    trg_file = 'data/english.en/english.en'


    # get dataset

    idx = 0
    dataset = NumberDataset(src_file, trg_file, zh_vocab, en_vocab, max_len)
    src_line = dataset.src_lines[idx]
    trg_line = dataset.trg_lines[idx]

    spacy_zh = spacy.load('zh_core_web_sm')
    spacy_en = spacy.load('en_core_web_sm')
    src_tokens = [tok.text for tok in spacy_zh(src_line)]
    trg_tokens = [tok.text for tok in spacy_en(trg_line)]
    
    src_nums = []
    trg_nums = []
    
    word = '1929年'
    print(f'{word} index is: {zh_vocab[word]}')
    
    src_tensor, trg_tensor = dataset[0]
    print(f"src tensor shape is: {src_tensor}")
    print(f"trg tensor shape is: {trg_tensor}")
    
def test3():
    zh_vocab_file_path = "zh_vocab.pkl"
    en_vocab_file_path = "en_vocab.pkl"

    with open(zh_vocab_file_path, "rb") as zh_vocab_file:
        zh_vocab = pickle.load(zh_vocab_file)

    with open(en_vocab_file_path, "rb") as en_vocab_file:
        en_vocab = pickle.load(en_vocab_file)
    print(f"len(zh_vocab) is: {len(zh_vocab)}")
    print(f"len(en_vocab) is: {len(en_vocab)}")
        
        
def test4():

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

    print(f"\
        src_vocab_size:{src_vocab_size}\n\
        trg_vocab_size:{trg_vocab_size}\n\
        src_pad_idx:{src_pad_idx}\n\
        trg_pad_idx:{trg_pad_idx}\n\
        embedding_size:{embedding_size}\n\
        num_layers:{num_layers}\n\
        num_heads:{num_heads}\n\
        dropout:{dropout}\n\
        forward_expansion:{forward_expansion}\n\
        DEVICE:{DEVICE} \n\
        max_len:{max_len}\n\
        ")
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
    
    
if __name__ == "__main__":
    test4()
