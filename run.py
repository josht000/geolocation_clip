import os
import torch
import argparse
import logging
from multiprocessing import set_start_method
from transformers import TrainingArguments
from src.train_modes import pretrain
from src.datasets import PretrainDatasetOSVMini

logger = logging.getLogger('run')

# Pretrain arguments for PIGEOTTO --> RUN ON 4 A100 GPUs
PRETAIN_ARGS = TrainingArguments(
        output_dir='saved_models/pretrained_osv-mini',
        overwrite_output_dir = True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8, # 12 for 3 GPUs
        learning_rate=5e-07, # was 1e-06 before
        weight_decay=0.001, # CHANGED
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-06,
        max_grad_norm=1.0,
        num_train_epochs=4, # 20 before
        max_steps=-1,
        lr_scheduler_type = 'linear',
        warmup_ratio = 0.02,
        logging_first_step = False,
        logging_steps=1,
        save_strategy='steps',
        save_steps=50,
        seed=42,
        dataloader_drop_last=True,
        run_name=None,
        adafactor=False,
        report_to='tensorboard',
        skip_memory_metrics=True,
        resume_from_checkpoint=None,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, default='pretrain', choices=['pretrain'])
    args = parser.parse_args()
    return args

def main():
    args = parse_args() 

    if args.function == 'pretrain':
        train_ds = PretrainDatasetOSVMini(split='train', dir='datasets/osv-mini-129k')
        test_ds = PretrainDatasetOSVMini(split='test', dir='datasets/osv-mini-129k')
        pretrain('openai/clip-vit-base-patch32', train_ds, test_ds, PRETAIN_ARGS, True)
    else:
        raise NotImplementedError(f'Mode {args.function} is not implemented.')

if __name__ == '__main__':
    set_start_method('spawn')
    main()
    
    

    
    