
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import prepare_datasets, pad_to_longest
from model import AdapterModel
from torch.optim import Adam, SGD
import random
import copy
import torch.autograd as auto
import os
from tqdm import tqdm
import time
import math

# import numpy as np
# For saving numpy rng
# not sure if this is needed
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray])

device_name = 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'
device = torch.device(device_name)

batch_size = 32

cache_dir = '/home/pipoika3/nobackup/autodelete/temp/'
cache_dir += f'maml-lora-{random.randint(0, 10**9-1):09d}'

if not os.path.exists(cache_dir): os.makedirs(cache_dir)

print('Using cache location', cache_dir)

model = AdapterModel()

get_train_set, val_sets, test_sets = prepare_datasets()

def run_batch_get_loss(model, batch):
    iid, imask, diid, dmask, ilang, dlang = batch
    
    iid = iid.to(device)
    imask = imask.to(device)
    diid = diid.to(device)
    dmask = dmask.to(device)

    labels = diid[:,1:]
    diid = diid[:,:-1]
    dmask = dmask[:,:-1]

    preds = model(iid, imask, diid, dmask, ilang, dlang)
    loss = criterion(preds.reshape(-1, preds.size(-1)), labels.reshape(-1))

    return loss

def infinite_dataloader(dataset):
    while True:
        yield from DataLoader(dataset, batch_size=batch_size, collate_fn=pad_to_longest)

def safe_add(a, b):
    if a is None: return b
    if b is None: return a
    return a+b

outer_optim = Adam(model.outer_params(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=1)

model_suffix = 'base'

# resume from checkpoint
outer_step = 0
best_val = math.inf
if os.path.exists(f'checkpoint_{model_suffix}.pt'):
    outer_state = torch.load(f'checkpoint_{model_suffix}.pt', weights_only=True, map_location=device)
    outer_step = outer_state['outer_step']
    best_val = outer_state['best_val']
    model.load_state_dict(outer_state['model_state'])
    outer_optim.load_state_dict(outer_state['optim_state'])

model.to(device)

for outer_step in range(outer_step, 10):

    num_tasks = 3 # was 2
    approx_level = 0 # change this

    outer_grads = None
    approx_grads = None

    # Tasks could be split across GPUs with Lightning
    for task_num in range(num_tasks):
        model.reset_adapter()

        cur_data = get_train_set()

        #### RUN SUPPORT SET
        inner_optim = SGD(model.inner_params(), lr=1e-3)

        support_loader = infinite_dataloader(cur_data)

        model.freeze_all()
        model.unfreeze_lora()
        # zero out all grads for safety
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()

        model.train()
        num_adapt_steps = len(support_loader) # grow this with the number of outer steps
        bar = tqdm(range(num_adapt_steps), desc=f'Outer {outer_step} Task {task_num} Forward Pass Loss ----')
        for inner_step in bar:
            batch = next(support_loader)

            # Could do grad accum here

            cur_state = {
                'batch': batch,
                'inner_state': model.inner_state_dict(),
                'inner_optim': inner_optim.state_dict(),
                'rng': {
                    'torch': torch.get_rng_state(),
                    # 'numpy': np.random.get_state(),
                    'random': random.getstate()
                }
            }
            path = os.path.join(cache_dir, f'{inner_step}.pt')
            torch.save(cur_state, path)

            loss = run_batch_get_loss(model, batch)
            bar.set_description(f'Outer {outer_step} Task {task_num} Forward Pass Loss {loss.item():.4f}')

            loss.backward()
            inner_optim.step()
            inner_optim.zero_grad()
        bar.close()


        #### ACCUMULATE QUERY SET
        cur_data.query_mode = True
        query_loader = DataLoader(cur_data, batch_size=batch_size, collate_fn=pad_to_longest)

        model.unfreeze_all()
        model.eval()
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

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        
        # setup outer and inner grads
        if outer_grads is None:
            outer_grads = [x.grad for x in model.outer_params()]
            outer_grads = [None if x is None else x.clone().detach() for x in outer_grads]
            approx_grads = outer_grads
        else:
            with torch.no_grad():
                # TODO: None checking
                outer_grads = [safe_add(a,b.grad) for a, b in zip(outer_grads, model.outer_params())]
                approx_grads = [safe_add(a,b.grad) for a, b in zip(approx_grads, model.outer_params())]

        # IF SIMPLE, continue here

        inner_grads = [x.grad.clone().detach() for x in model.inner_params()]

        #### BACKPROP TROUGH TRAINING

        model.train()
        # Vanishing gradient here???, consider only doing a few steps to minimize it work needed
        for inner_step in tqdm(range(num_adapt_steps), desc=f'Outer {outer_step} Task {task_num} Backward Pass'):
            path = os.path.join(cache_dir, f'{num_adapt_steps-1-inner_step}.pt')
            checkpoint = torch.load(path, weights_only=True)

            batch = checkpoint['batch']
            torch.set_rng_state(checkpoint['rng']['torch'])
            random.setstate(checkpoint['rng']['random'])

            inner_state_dict = checkpoint['inner_state']
            model.load_state_dict(inner_state_dict, strict=False)

            # zero grad here
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # this is kind of slow
            # I could just copy once if I can replace the past inner values with leaf nodes
            next_model = copy.deepcopy(model)
            inner_optim = SGD(next_model.inner_params(), lr=1e-3, differentiable=True)
            inner_optim.load_state_dict(checkpoint['inner_optim'])

            model.unfreeze_all()
            next_model.freeze_all()

            loss = run_batch_get_loss(model, batch)

            # compute the higher order gradient here
            grads = auto.grad(loss, model.inner_params(), create_graph=True, retain_graph=True)

            for p, g in zip(next_model.inner_params(), grads):
                p.grad = g

            inner_optim.step()

            # only backward once is faster
            outer_loss = 0
            for p, g in zip(next_model.inner_params(), inner_grads):
                # p.backward(g, retain_graph=True)
                outer_loss += (g.detach() * p).sum()
            outer_loss.backward()

            # update inner / outer grads here
            # print(model.outer_params()[0].grad)
            # print(model.outer_params()[1].grad[:5,:5])

            # print(model.inner_params()[0].grad)
            # print(model.inner_params()[1].grad[:5,:5])

            inner_grads = [x.grad.clone().detach() for x in model.inner_params()]

            with torch.no_grad():
                outer_grads = [safe_add(a,b.grad) for a, b in zip(outer_grads, model.outer_params())]

            if inner_step < approx_level:
                with torch.no_grad():
                    approx_grads = [safe_add(a,b.grad) for a, b in zip(approx_grads, model.outer_params())]

            # Maybe break early if inner_grads has really small norm


    if outer_grads is not None:
        # TODO: consider computing angle between full and approx here

        with torch.no_grad():
            for p, g in zip(model.outer_params(), outer_grads):
                p.grad = g / num_tasks if g is not None else g

        outer_optim.step()
        outer_optim.zero_grad() # not really needed, but doesn't hurt to be safe

    ### VALIDATION
    val_perf = 0
    for i, cur_data in enumerate(val_sets):
        model.reset_adapter()

        #### RUN SUPPORT SET
        inner_optim = SGD(model.inner_params(), lr=1e-3)

        # TODO: shuffling here can make it non deterministic, but we do want to shuffle across epochs
        support_loader = infinite_dataloader(cur_data)

        model.freeze_all()
        model.unfreeze_lora()
        # zero out all grads for safety
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()

        model.train()
        num_adapt_steps = len(support_loader)
        bar = tqdm(range(num_adapt_steps), desc=f'Outer {outer_step} Validation {i+1}/{len(val_sets)} Forward Pass Loss ----')
        for inner_step in bar:
            batch = next(support_loader)

            loss = run_batch_get_loss(model, batch)
            bar.set_description(f'Outer {outer_step} Validation {i+1}/{len(val_sets)} Forward Pass Loss {loss.item():.4f}')

            loss.backward()
            inner_optim.step()
            inner_optim.zero_grad()
        bar.close()

        #### EVALUATE QUERY SET
        cur_data.query_mode = True
        query_loader = DataLoader(cur_data, batch_size=batch_size, collate_fn=pad_to_longest)

        with torch.no_grad():
            model.eval()

            bar = tqdm(total=len(query_loader), desc=f'Outer {outer_step} Validation {i+1}/{len(val_sets)} Query Loss ----')
            total_loss = 0
            total_steps = 0
            for batch in query_loader:
                my_batch = len(batch[-1])
                loss = run_batch_get_loss(model, batch) * my_batch
                total_loss += loss.item()
                total_steps += my_batch
                bar.set_description(f'Outer {outer_step} Validation {i+1}/{len(val_sets)} Query Loss {total_loss / total_steps:.4f}')
                bar.update(1)
            bar.close()
            val_perf += total_loss / total_steps
    val_perf /= len(val_sets)

    if val_perf < best_val:
        best_val = val_perf
        print(f'Best performance so far {best_val}')
        torch.save(model.state_dict(), f'best_model_{model_suffix}.pt')


    # save a checkpoint now
    
    outer_state = {
        'outer_step': outer_step + 1,
        'best_val': best_val,
        'model_state': model.state_dict(),
        'optim_state': outer_optim.state_dict()
    }

    torch.save(outer_state, f'checkpoint_{model_suffix}.pt')


# load best model
model.load_state_dict(torch.load(f'best_model_{model_suffix}.pt', weights_only=True, map_location=device))


### RUNNING TEST SET
test_loss_perf = 0
for i, cur_data in enumerate(test_sets):
    model.reset_adapter()

    #### RUN SUPPORT SET
    inner_optim = SGD(model.inner_params(), lr=1e-3)

    # TODO: shuffling here can make it non deterministic, but we do want to shuffle across epochs
    support_loader = infinite_dataloader(cur_data)

    model.freeze_all()
    model.unfreeze_lora()
    # zero out all grads for safety
    for p in model.parameters():
        if p.grad is not None: p.grad.zero_()

    model.train()
    num_adapt_steps = len(support_loader)
    bar = tqdm(range(num_adapt_steps), desc=f'Test Set {i+1}/{len(val_sets)} Forward Pass Loss ----')
    for inner_step in bar:
        batch = next(support_loader)

        loss = run_batch_get_loss(model, batch)
        bar.set_description(f'Test Set {i+1}/{len(val_sets)} Forward Pass Loss {loss.item():.4f}')

        loss.backward()
        inner_optim.step()
        inner_optim.zero_grad()
    bar.close()

    #### EVALUATE QUERY SET
    cur_data.query_mode = True
    query_loader = DataLoader(cur_data, batch_size=batch_size, collate_fn=pad_to_longest)

    with torch.no_grad():
        model.eval()

        bar = tqdm(total=len(query_loader), desc=f'Test Set {i+1}/{len(val_sets)} Query Loss ----')
        total_loss = 0
        total_steps = 0
        for batch in query_loader:
            my_batch = len(batch[-1])
            loss = run_batch_get_loss(model, batch) * my_batch
            total_loss += loss.item()
            total_steps += my_batch

            # TODO: run generation

            bar.set_description(f'Test Set {i+1}/{len(val_sets)} Query Loss {total_loss / total_steps:.4f}')
            bar.update(1)
        bar.close()
        val_perf += total_loss / total_steps
test_loss_perf /= len(test_sets)


for _ in range(10): print()
print('Average Test Set Loss')
print(test_loss_perf)