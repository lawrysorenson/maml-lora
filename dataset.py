
import sentencepiece as spm
from torch.utils.data import Dataset
import random
import os
from tqdm import tqdm
import io

# TODO: on the fly tokenization
"""
spm.SentencePieceTrainer.train(
    input='./langs/Spanish/filtered/src.txt',
    model_prefix='toks/eng',
    vocab_size=16000,
    user_defined_symbols=['<lang>', '<train>', '<eval>'],
    unk_id=0,
    bos_id=-1,
    pad_id=1,
    eos_id=2
)
"""

data_dir = '/home/lawry/cs674/data/'

def train_spm(sents):
    vocab_size = 8000

    while vocab_size >= 500:
        model = io.BytesIO()
        try:            
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(sents),
                model_writer=model,
                vocab_size=vocab_size,
                user_defined_symbols=['<lang>'],
                unk_id=0, # potentially disable more of these, right now is consistent with English usage
                bos_id=-1,
                pad_id=1,
                eos_id=2
            )
        except Exception as e:
            if 'Vocabulary size too high':
                vocab_size //= 2
                continue
            else: raise e

        ans = spm.SentencePieceProcessor(model_proto=model.getvalue())

        sent = sents[0]

        print(sent)
        print(ans.encode(sent))

        return ans
    
    raise Exception('Vocab too small')

class MAMLDataset(Dataset):
    def __init__(self, eng_tok, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.eng_tok = eng_tok
        self.tgt_tok = train_spm([s[1] for s in train_data]) # CREATE SPM ON THE FLY
        self.test_mode = False
        self.dir = -1

    def __len__(self):
        data = [self.train_data, self.test_data][self.test_mode]
        return len(data)
    
    def __getitem__(self, i):
        data = [self.train_data, self.test_data][self.test_mode]

        atok = self.eng_tok
        btok = self.tgt_tok
        a, b = data[i]

        a = atok.encode(a)
        b = btok.encode(b)

        if self.dir==1 or (self.dir == -1 and random.random() < 0.5):
            a,b = b,a

        print(a, b)

        # say direction

        # b negative?
        

def load_tok(lang):
    path = os.path.join(data_dir, 'toks', f'{lang}.model')
    return spm.SentencePieceProcessor(model_file=path)


def load_segments(lang):
    path = os.path.join(data_dir, 'langs', lang, 'filtered')
    with open(os.path.join(path, 'src.txt'), 'r') as sfile, \
        open(os.path.join(path, 'tgt.txt')) as tfile:
        ans = list(zip(sfile, tfile))
        ans = [(a.strip(), b.strip()) for a, b in ans]
        return ans
            


def load_language_data():

    eng_tok = load_tok('eng')

    fold = os.path.join(data_dir, 'langs')

    ans = []
    for lang in tqdm(sorted(os.listdir(fold)), desc='Loading data'):
        segs = load_segments(lang)

        segs = segs[:10000] # REMOVE

        ans += [segs]

        if len(ans) == 10: break  # REMOVE

    return eng_tok, ans


test_size = 5000
train_size = 10000
train_test_split = 0.8
def prep_maml_test(eng_tok, langs):
    ans = []
    for segs in langs:
        for _ in range(5):
            random.shuffle(segs)
            split = int(len(segs) * train_test_split)
            train_data = segs[:split][:train_size]
            test_data = segs[split:][:test_size]

            sub = MAMLDataset(eng_tok, train_data, test_data)
            for a, b, d in sub:
                print(a, b, d)
            exit()
            ans += [sub]
    return ans

def prepare_datasets():
    
    eng_tok, langs = load_language_data()

    print(dir(eng_tok))
    # print(eng_tok.piece_to_id('<lang>'))
    # print(eng_tok.eos_id())
    # print(eng_tok.bos_id()) # use pad = bos
    # print(eng_tok.pad_id())

    random.shuffle(langs)
     
    n = len(langs)

    train_cut = int(n*0.8)
    val_cut = int(n*0.9)

    train_langs = langs[:train_cut]
    val_langs = langs[train_cut:val_cut]
    test_langs = langs[val_cut:]

    val_set = prep_maml_test(eng_tok, val_langs)




if __name__ == '__main__':
    prepare_datasets()