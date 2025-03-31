import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from typing import Any, Dict, Union
import os

from transformers import Trainer, TrainingArguments, \
                         AutoModelForImageClassification, \
                         CLIPVisionModel, CLIPModel, CLIPProcessor
from warnings import filterwarnings

filterwarnings('ignore', category=UserWarning, module='transformers')

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')

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
        train_ds (PretrainDataset): Dataset to be used for contrastive pretraining.
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
    print('Before training: Accuracy on batch size of 16 is', before_acc)
    
    # Train
    if resume:
        print('Resuming training from checkpoint ...')
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    
    # After training
    after_acc = test_ds.accuracy(model, batch_size=16)
    print('After training: Accuracy on batch size of 16 is', after_acc)


def geo_collate_fn(examples):
    """Collate function for CLIPGeolocationDataset batch processing."""
    batch = {}
    
    # Only include keys that are compatible with CLIPVisionModel
    # Specifically exclude text-related inputs like input_ids and attention_mask
    vision_model_keys = ['pixel_values', 'climate_labels', 'month_labels', 'state_labels', 
                         'county_labels', 'city_labels', 'lat_labels', 'lng_labels']
    
    for k in examples[0].keys():
        if k in vision_model_keys:
            batch[k] = torch.stack([example[k] for example in examples])
    
    return batch

def get_versioned_dir(base_output_dir):
    """Create a versioned output directory if the base directory already exists.
    
    Args:
        base_output_dir (str): The base output directory path
        
    Returns:
        str: A versioned output directory path
    """
    if os.path.exists(base_output_dir):
        # Find existing versioned directories
        dir_name = os.path.basename(base_output_dir)
        parent_dir = os.path.dirname(base_output_dir)
        version_dirs = [d for d in os.listdir(parent_dir) 
                       if d.startswith(dir_name + "_v")]
        
        # Extract version numbers
        existing_versions = [int(d.split("_v")[-1]) for d in version_dirs if d.split("_v")[-1].isdigit()]
        
        # Calculate new version number
        new_version = 1
        if existing_versions:
            new_version = max(existing_versions) + 1
            
        # Create new versioned directory
        versioned_output_dir = f"{base_output_dir}_v{new_version}"
        logger.info(f"Output directory {base_output_dir} already exists. Creating {versioned_output_dir}")
        os.makedirs(versioned_output_dir, exist_ok=True)
        return versioned_output_dir
    
    # If directory doesn't exist, create it and return the original path
    os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir

def train_geolocation_model(
    model: Union[CLIPModel, str],
    train_dataset: Any,
    val_dataset: Any,
    training_args: TrainingArguments,
    device: str = 'cuda'
) -> CLIPModel:
    """Train a geolocation model based on CLIP architecture.
    
    Args:
        model: Model to train or path to pretrained model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_args: HuggingFace TrainingArguments
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained model
    """
    # Version the output directory
    training_args.output_dir = get_versioned_dir(training_args.output_dir)
    
    # Load model if path is provided
    if isinstance(model, str):
        model = CLIPModel.from_pretrained(model)
    
    model.to(device)
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=geo_collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=geo_collate_fn,
        num_workers=2
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * training_args.num_train_epochs // training_args.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps
    )
    
    # Tensorboard
    writer = SummaryWriter(log_dir=f"{training_args.output_dir}/logs")
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{int(training_args.num_train_epochs)}")
        for step, batch in enumerate(progress_bar):
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Backward pass with gradient accumulation
            loss = loss / training_args.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Log training metrics
                global_step += 1
                epoch_loss += loss.item() * training_args.gradient_accumulation_steps
                progress_bar.set_postfix({"loss": loss.item() * training_args.gradient_accumulation_steps})
                
                if training_args.logging_steps > 0 and global_step % training_args.logging_steps == 0:
                    writer.add_scalar("train/loss", loss.item() * training_args.gradient_accumulation_steps, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                
                # Evaluation
                should_evaluate = (
                    training_args.eval_strategy == "steps" and 
                    training_args.eval_steps > 0 and 
                    global_step % training_args.eval_steps == 0
                )
                
                if should_evaluate:
                    val_metrics = evaluate_geolocation_model(model, val_loader, device)
                    for metric_name, metric_value in val_metrics.items():
                        writer.add_scalar(f"eval/{metric_name}", metric_value, global_step)
                    
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        logger.info(f"New best model with validation loss: {best_val_loss:.4f}")
                        model.save_pretrained(f"{training_args.output_dir}/best_model")
                
                # Save checkpoint
                should_save = (
                    training_args.save_strategy == "steps" and 
                    training_args.save_steps > 0 and 
                    global_step % training_args.save_steps == 0
                )
                
                if should_save:
                    model.save_pretrained(f"{training_args.output_dir}/checkpoint-{global_step}")
        
        # Epoch completed
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Full evaluation at end of epoch
        should_evaluate_epoch = (
            training_args.eval_strategy == "epoch"
        )
        
        if should_evaluate_epoch:
            val_metrics = evaluate_geolocation_model(model, val_loader, device)
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f"eval/{metric_name}", metric_value, global_step)
                logger.info(f"Validation {metric_name}: {metric_value:.4f}")
        
        # Save epoch checkpoint
        should_save_epoch = (
            training_args.save_strategy == "epoch"
        )
        
        if should_save_epoch:
            model.save_pretrained(f"{training_args.output_dir}/checkpoint-epoch-{epoch+1}")
    
    writer.close()
    return model

def evaluate_geolocation_model(
    model: Union[CLIPModel, str],
    val_loader: DataLoader = None,
    device: str = 'cuda',
    val_dataset: Any = None,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate a geolocation model.
    
    Args:
        model: Model to evaluate or path to model
        val_loader: Validation data loader (if None, one will be created from val_dataset)
        device: Device to evaluate on
        val_dataset: Validation dataset (used if val_loader is None)
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model if path is provided
    if isinstance(model, str):
        model = CLIPModel.from_pretrained(model)
        model.to(device)
    
    # Create validation loader if not provided
    if val_loader is None:
        assert val_dataset is not None, "Either val_loader or val_dataset must be provided"
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=geo_collate_fn,
            num_workers=2
        )
    
    # Set model to evaluation mode
    model.eval()
    
    # Track metrics
    total_loss = 0
    total_climate_loss = 0
    total_month_loss = 0
    total_location_loss = 0
    total_distance_loss = 0
    
    total_correct_climate = 0
    total_correct_month = 0
    total_correct_state = 0
    total_correct_county = 0
    total_correct_city = 0
    total_samples = 0
    
    total_distance_error = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**inputs, is_training=False)
            
            # Track losses
            total_loss += outputs.loss.item() * len(batch)
            total_climate_loss += outputs.loss_climate.item() * len(batch)
            total_month_loss += outputs.loss_month.item() * len(batch)
            total_location_loss += outputs.loss_location.item() * len(batch)
            total_distance_loss += outputs.loss_distance.item() * len(batch)
            
            # Track accuracies
            climate_preds = outputs.preds_climate.argmax(dim=1)
            month_preds = outputs.preds_month.argmax(dim=1)
            state_preds = outputs.preds_state.argmax(dim=1)
            county_preds = outputs.preds_county.argmax(dim=1)
            city_preds = outputs.preds_city.argmax(dim=1)
            
            total_correct_climate += (climate_preds == inputs['climate_labels']).sum().item()
            total_correct_month += (month_preds == inputs['month_labels']).sum().item()
            total_correct_state += (state_preds == inputs['state_labels']).sum().item()
            total_correct_county += (county_preds == inputs['county_labels']).sum().item()
            total_correct_city += (city_preds == inputs['city_labels']).sum().item()
            
            # Track location error (distance in km)
            # This is already computed in the model as part of loss_distance
            
            total_samples += len(batch)
    
    # Calculate metrics
    metrics = {
        "loss": total_loss / total_samples,
        "climate_loss": total_climate_loss / total_samples,
        "month_loss": total_month_loss / total_samples,
        "location_loss": total_location_loss / total_samples,
        "distance_loss": total_distance_loss / total_samples,
        "climate_accuracy": total_correct_climate / total_samples,
        "month_accuracy": total_correct_month / total_samples,
        "state_accuracy": total_correct_state / total_samples,
        "county_accuracy": total_correct_county / total_samples,
        "city_accuracy": total_correct_city / total_samples,
        "average_distance_km": total_distance_loss / total_samples / model.DISTANCE_LOSS_SCALING
    }
    
    return metrics

def profile_model_performance(
    model_path: str,
    dataset: Any,
    batch_size: int = 16,
    num_steps: int = 20,
    device: str = 'cuda',
    profile_output: str = 'profile_results'
):
    """Profile the performance of a geolocation model.
    
    Args:
        model_path: Path to pretrained model
        dataset: Dataset for profiling
        batch_size: Batch size for profiling
        num_steps: Number of steps to profile
        device: Device to profile on
        profile_output: Output directory for profile results
    """
    import torch.profiler
    
    # Load model
    model = CLIPModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Create dataloader with limited steps
    limited_dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), batch_size * num_steps)))
    dataloader = DataLoader(
        limited_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=geo_collate_fn,
        num_workers=2
    )
    
    # Setup profiler
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=num_steps,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_output),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    
    # Profile forward pass
    logger.info(f"Profiling model for {num_steps} steps with batch size {batch_size}")
    
    with profiler:
        for step, batch in enumerate(tqdm(dataloader, desc="Profiling")):
            if step >= num_steps:
                break
                
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**inputs)
            
            # Record step
            profiler.step()
    
    logger.info(f"Profiling complete. Results saved to {profile_output}")
    
    # Print key stats
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
