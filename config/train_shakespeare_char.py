# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 2
batch_size = 40
block_size = 512 # context of up to 512 previous characters

n_layer = 10
n_head = 16
n_embd = 512
dropout = 0.15

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 1500
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

weight_decay = 1e-1



# next token baseline: 
# step 1000: train loss 1.0870, val loss 1.4768 (lr 1e-3, dropout 0.3) * 

# step 1500: train loss 2.1345, val loss 2.3614 (lr 1e-5, dropout 0.2)
# step 1500: train loss 1.7811, val loss 2.1740 (lr 1e-3, dropout 0.4)
# step 1500: train loss 1.9536, val loss 2.2579 (lr 1e-3, dropout 0.5)
# step 750 : train loss 1.7220, val loss 2.1908 (lr 1e-3, dropout 0.1, overfitted after 750)
# step 1500: train loss 1.6305, val loss 2.1660 (lr 1e-3, dropout 0.3) * 
# step 1500: train loss 1.6350, val loss 2.1830 (lr 1e-3, dropout 0.3, block_size 512))
# step 750: train loss 1.7841, val loss 2.1851 (lr 1e-3, dropout 0.15, block_size 512, overfitted after 750))
