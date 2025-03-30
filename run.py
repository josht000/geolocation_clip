import os
import torch
import argparse
import logging
from multiprocessing import set_start_method
from transformers import TrainingArguments
from src.train_modes import pretrain, train_geolocation_model
from src.datasets import PretrainDatasetOSVMini, CLIPGeolocationDataset
from src.models.clip_model import CLIPModel
from warnings import filterwarnings

filterwarnings('ignore', category=UserWarning, module='transformers')
filterwarnings('ignore', category=FutureWarning, module='transformers')

logger = logging.getLogger('run')

# Pretrain arguments for PIGEOTTO --> RUN ON 4 A100 GPUs
PRETAIN_ARGS = TrainingArguments(
        output_dir='saved_models/pretrained_osv-mini',
        overwrite_output_dir = True,
        do_train=True,
        do_eval=True,
        eval_strategy='steps',
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

GEO_TRAIN_ARGS = TrainingArguments(
    output_dir='saved_models/geolocation_model',
    remove_unused_columns=False,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    eval_strategy='steps',  # Updated from evaluation_strategy
    eval_steps=500,
    save_strategy='steps',
    save_steps=1000,
    learning_rate=2e-5,
    logging_steps=50,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    seed=330
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, default='pretrain', choices=['pretrain', 'train'])
    args = parser.parse_args()
    return args

def main():
    args = parse_args() 

    if args.function == 'pretrain':
        train_ds = PretrainDatasetOSVMini(split='train', dir='datasets/osv-mini-129k')
        test_ds = PretrainDatasetOSVMini(split='test', dir='datasets/osv-mini-129k')
        pretrain('openai/clip-vit-base-patch32', train_ds, test_ds, PRETAIN_ARGS, True)
    elif args.function == 'train':
        train_ds = CLIPGeolocationDataset('train', 'datasets/osv-mini-129k', 
                                          'datasets/unique_city_list.txt', 
                                          'datasets/unique_sub-region_list.txt', True, 224)
        test_ds = CLIPGeolocationDataset('test', 'datasets/osv-mini-129k', 
                                         'datasets/unique_city_list.txt', 
                                         'datasets/unique_sub-region_list.txt', False, 224)
        
        model = CLIPModel('saved_models/checkpoint-1400')
        # Use TrainingArguments directly
        trained_model = train_geolocation_model(
            model, 
            train_ds, 
            test_ds, 
            GEO_TRAIN_ARGS,
            'cuda'
        )
    else:
        raise NotImplementedError(f'Mode {args.function} is not implemented.')

if __name__ == '__main__':
    set_start_method('spawn')
    main()
