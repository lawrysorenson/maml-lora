
import torch
import torch.nn as nn
from dataset import prepare_datasets, pad_to_longest, DataLoader
from model import AdapterModel
from torch.optim import Adam


model = AdapterModel()

get_train_set, val_sets, test_sets = prepare_datasets()

optim = Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=1)

for iid, imask, diid, dmask, ilang, dlang in DataLoader(val_sets[0], batch_size=5, collate_fn=pad_to_longest):
    labels = diid[:,1:]
    diid = diid[:,:-1]
    dmask = dmask[:,:-1]

    for _ in range(2000):
        preds = model(iid, imask, diid, dmask, ilang, dlang)
        loss = criterion(preds.reshape(-1, preds.size(-1)), labels.reshape(-1))

        loss.backward()
        optim.step()
        optim.zero_grad()

        print(loss)
        print(preds[0].argmax(-1))
        print(labels[0])
        print()

    break