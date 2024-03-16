# config for training GPT-2 (124M)

# $ torchrun --standalone --nproc_per_node=2 train.py config/train_gpt2.py

wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-124M'


batch_size = 18
block_size = 1024
gradient_accumulation_steps = 2

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
