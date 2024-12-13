
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
from tqdm import tqdm

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

        #### RUN SUPPORT SET
        inner_optim = Adam(model.inner_params(), lr=3e-4)

        support_loader = infinite_dataloader(cur_data)

        model.freeze_all()
        model.unfreeze_lora()
        # zero out all grads for safety
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()

        for inner_step in tqdm(range(10), desc=f'Outer {outer_step} Task {task_num} Forward Pass'):
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


        #### ACCUMULATE QUERY SET
        cur_data.query_mode = True
        query_loader = DataLoader(cur_data, batch_size=batch_size, collate_fn=pad_to_longest)

        model.unfreeze_all()
        grad_scaler = len(cur_data)

        bar = tqdm(total=len(query_loader), desc=f'Outer {outer_step} Task {task_num} Accumulating Query Loss ----')
        total_loss = 0
        total_steps = 0
        for batch in query_loader:
            my_batch = len(batch[-1])
            loss = run_batch_get_loss(model, batch) * my_batch
            total_loss += loss.item()
            total_steps += my_batch
            (loss / grad_scaler).backward()
            bar.set_description(f'Outer {outer_step} Task {task_num} Accumulating Query Loss {total_loss / total_steps:.4f}')
            bar.update(1)
        bar.close()

        # setup outer and inner grads
        if outer_grads is None:
            outer_grads = [x.clone().detach() for x in model.outer_params()]
        else:
            with torch.no_grad():
                # TODO: None checking
                outer_grads = [a+b for a, b in zip(outer_grads, model.outer_params())]

        inner_grads = [x.clone().detach() for x in model.inner_params()]

        #### BACKPROP TROUGH TRAINING
        # do I want to differentiate Adam???
        inner_optim = Adam(model.inner_params(), lr=3e-4, differentiable=True)

        exit()



        # figure out backprop here
        break

    # TODO: scale by task num
    # outer_grads


    # TODO: validation, find best model
    break
    


# TODO: run test set