import logging
import torch

from transformers import Trainer, TrainingArguments, \
                         AutoModelForImageClassification, \
                         CLIPVisionModel, CLIPModel, CLIPProcessor

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')
BASE_CLIP_MODEL = 'openai/clip-vit-base-patch32'
clip_processor = CLIPProcessor.from_pretrained(BASE_CLIP_MODEL)

def collate_fn(examples):
    images = [example[0] for example in examples]
    text = [example[1] for example in examples]
    inputs = clip_processor(images=images, text=text, return_tensors='pt',
                            padding=True, truncation=True)
    inputs['return_loss'] = True
    return inputs

def pretrain(model: str, train_ds, test_ds, train_args: TrainingArguments, 
             resume: bool=False) -> CLIPModel:
    """Pretrains a CLIP model on the given dataset.

    Args:
        model (str): Name of Huggingface model or trainable object.
        train_ds (PretrainDataset): Dataset to be used for contrasrive pretraining.
        test_ds (PretrainDataset): Dataset to be used for evaluation.
        train_args (TrainingArguments, optional): Pretraining arguments. Defaults to PRETAIN_ARGS.
        resume (bool, optional): Whether to resume model training from checkpoint.
    Returns:
        CLIPModel: Pretrained CLIP model.
    """
    model = CLIPModel.from_pretrained(model)
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collate_fn,
    )
    
    # Before training
    before_acc = train_ds.accuracy(model, batch_size=16)
    print('Before traing: Accuracy on batch size of 16 is', before_acc)
    
    # Train
    if resume:
        print('Resuming training from checkpoint ...')
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    
    # After training
    after_acc = test_ds.accuracy(model, batch_size=16)
    print('After training: Accuracy on batch size of 16 is', after_acc)