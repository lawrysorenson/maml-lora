
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import prepare_datasets, pad_to_longest
from model import AdapterModel
from torch.optim import Adam
import random
import copy
import numpy as np
import os

batch_size = 5

model = AdapterModel()

get_train_set, val_sets, test_sets = prepare_datasets()

outer_optim = Adam(model.outer_params(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=1)

def run_batch_get_loss(model, batch):
    iid, imask, diid, dmask, ilang, dlang = batch
    labels = diid[:,1:]
    diid = diid[:,:-1]
    dmask = dmask[:,:-1]

    preds = model(iid, imask, diid, dmask, ilang, dlang)
    loss = criterion(preds.reshape(-1, preds.size(-1)), labels.reshape(-1))

    return loss

cache_dir = 'temp/'
cache_dir += f'maml-lora-{random.randint(0, 10**9-1):09d}'

if not os.path.exists(cache_dir): os.makedirs(cache_dir)

print(cache_dir)

def infinite_dataloader(dataset):
    while True:
        yield from DataLoader(dataset, batch_size=batch_size, collate_fn=pad_to_longest)

for outer_step in range(1):

    num_tasks = 10

    outer_grads = None

    for task_num in range(num_tasks):
        model.reset_adapter()

        cur_data = get_train_set()
        inner_optim = Adam(model.inner_params(), lr=3e-4)

        support_loader = infinite_dataloader(cur_data)

        model.freeze_all()
        model.unfreeze_lora()

        for inner_step in range(10):
            batch = next(support_loader)

            cur_state = {
                'batch': batch,
                'inner_state': model.inner_state_dict(),
                'inner_optim': inner_optim.state_dict(),
                'rng': {
                    'torch': torch.get_rng_state(),
                    'numpy': np.random.get_state(),
                    'random': random.getstate()
                }
            }
            path = os.path.join(cache_dir, f'{inner_step}.pt')
            torch.save(cur_state, path)

            loss = run_batch_get_loss(model, batch)

            loss.backward()
            inner_optim.step()
            inner_optim.zero_grad()



        # do I want to differentiate Adam???
        cur_data.query_mode = True
        query_loader = DataLoader(cur_data, batch_size=batch_size, collate_fn=pad_to_longest)

        # accumulate full query set here


        inner_optim = Adam(model.inner_params(), lr=3e-4, differentiable=True)

        exit()



        # figure out backprop here
        break

    # TODO: validation, find best model
    break

# for iid, imask, diid, dmask, ilang, dlang in :

    break


# run test set