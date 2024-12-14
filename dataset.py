
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
import torch
import random
import os
import sys
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

# data_dir = '/home/lawry/cs674/data/'
data_dir = '/home/pipoika3/nobackup/autodelete/church-data/'

def train_spm(sents):
    vocab_size = 8000

    n = len(sents)
    # don't use all of the train set in the tokenizer
    # this ensures that the unk token will be trained
    sents = sents[:int(n*0.95)]

    while vocab_size >= 50: # 500
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
                eos_id=2,
                minloglevel=100,
                num_threads=4
            )
        except Exception as e:
            if 'Vocabulary size too high':
                vocab_size //= 2
                continue
            else: raise e

        ans = spm.SentencePieceProcessor(model_proto=model.getvalue())

        sent = sents[0]

        return ans
    
    raise Exception('Vocab too small')

class MAMLDataset(Dataset):
    def __init__(self, eng_tok, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.eng_tok = eng_tok
        self.tgt_tok = train_spm([s[1] for s in train_data]) # CREATE SPM ON THE FLY
        self.query_mode = False
        self.dir = 1 # other to English
        self.src_lang = 'eng'
        self.tgt_lang = 'oth' # placeholder

    def __len__(self):
        data = [self.train_data, self.test_data][self.query_mode]
        return len(data)
    
    def __getitem__(self, i):
        data = [self.train_data, self.test_data][self.query_mode]

        atok = self.eng_tok
        btok = self.tgt_tok
        a, b = data[i]

        # TODO: SPM SAMPLING
        a = atok.encode(a)
        b = [t for t in btok.encode(b)]

        eng_lang = atok.piece_to_id('<lang>')
        tgt_lang = btok.piece_to_id('<lang>')

        if self.dir==1 or (self.dir == -1 and random.random() < 0.5):
            
            # flipped direction            
            return [tgt_lang] + b, [eng_lang] + a + [atok.eos_id()], self.tgt_lang, self.src_lang
        else:
            # normal direction
            return [eng_lang] + a, [tgt_lang] + b + [btok.eos_id()], self.src_lang, self.tgt_lang


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

        # segs = segs[:10000] # REMOVE

        ans += [segs]

        # if len(ans) == 10: break  # REMOVE

    return eng_tok, ans


test_size = 5000
train_size = 10000
train_test_split = 0.8
num_per_lang = 2
def prep_maml_test(eng_tok, langs, desc):
    ans = []

    bar = tqdm(total=num_per_lang*len(langs), desc=desc)
    for segs in langs:
        for _ in range(num_per_lang):
            random.shuffle(segs)
            split = int(len(segs) * train_test_split)
            train_data = segs[:split][:train_size]
            test_data = segs[split:][:test_size]

            sub = MAMLDataset(eng_tok, train_data, test_data)
            ans += [sub]
            bar.update(1)
            # break # REMOVE THIS
    bar.close()
    return ans

PAD_ID = 1
def _pad_helper(data):
  sizes = [len(row) for row in data]

  max_size = max(sizes)

  for row in data:
    row.extend([PAD_ID]*(max_size - len(row)))

  data = torch.tensor(data)

  mask = torch.ones(data.size())
  for i,s in enumerate(sizes): mask[i,s:] = 0

  return data, mask

def pad_to_longest(batch):
    srcs, tgts, ins, outs = zip(*batch)

    return *_pad_helper(srcs), *_pad_helper(tgts), ins, outs

def prepare_datasets():
    
    eng_tok, langs = load_language_data()

    random.seed(42)
    random.shuffle(langs)
     
    n = len(langs)

    train_cut = int(n*0.8)
    val_cut = int(n*0.9)

    train_langs = langs[:train_cut]
    val_langs = langs[train_cut:val_cut]
    test_langs = langs[val_cut:]

    val_set = prep_maml_test(eng_tok, val_langs, 'Preparing val set')
    test_set = prep_maml_test(eng_tok, test_langs, 'Preparing test set')

    def get_train_set(): # support and query are intermixed???
        segs = random.choice(train_langs)
        random.shuffle(segs)
        split = int(len(segs) * train_test_split)
        train_data = segs[:split][:train_size]
        test_data = segs[split:][:test_size]

        sub = MAMLDataset(eng_tok, train_data, test_data)
        return sub
        

    return get_train_set, val_set, test_set




if __name__ == '__main__':
    a, b, c = prepare_datasets()

    for x, y, z in a():
        print(x, y, z)
        break

    # for a, am, b, bm, d in DataLoader(val_set[0], batch_size=5, collate_fn=pad_to_longest):
    #     print(a, am, b, bm, d)
    #     break