import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import CLIPModel
from typing import Any, Dict, Union
import os

from transformers import Trainer, TrainingArguments, CLIPModel
from warnings import filterwarnings

from src.constants import *
from src.models.train_losses import GeoCLIPLoss, l2_distance_tensor, calculate_batch_metrics

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
    
    # Get all keys from the first example
    available_keys = examples[0].keys()
    
    # Only include keys that are available in the examples
    for k in available_keys:
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
    device: str = 'cuda',
    use_context: bool = True
) -> CLIPModel:
    """Train a geolocation model based on CLIP architecture.
    
    Args:
        model: Model to train or path to pretrained model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        training_args: HuggingFace TrainingArguments
        device: Device to train on ('cuda' or 'cpu')
        use_context: Whether to use contextual features during training
        
    Returns:
        Trained model
    """
    # Version the output directory
    training_args.output_dir = get_versioned_dir(training_args.output_dir)
    
    # Load model if path is provided
    if isinstance(model, str):
        model = CLIPModel.from_pretrained(model)
    
    model = model.to(device)
    model.train()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=geo_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=geo_collate_fn
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    
    # Initialize learning rate scheduler
    num_training_steps = len(train_loader) * training_args.num_train_epochs
    total_steps = int(num_training_steps * training_args.warmup_ratio)
    
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_steps
    )
    
    # Initialize TensorBoard
    writer = SummaryWriter(training_args.output_dir)
    
    # Initialize loss function
    loss_fn = GeoCLIPLoss(device)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    print(f'Training for {training_args.num_train_epochs} epochs...')

    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        total_loss = 0
        total_climate_loss = 0
        total_month_loss = 0
        total_location_loss = 0
        total_distance_loss = 0
        
        # Metrics for location accuracy
        correct_state = 0
        correct_county = 0
        correct_city = 0
        total_samples = 0
        
        # Metrics for coordinate prediction
        total_distance_error = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=80)):
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(inputs['pixel_values'])
            
            # Calculate loss using GeoCLIPLoss
            loss, loss_dict = loss_fn(
                outputs,
                climate_labels=inputs.get('climate_labels') if use_context and 'climate_labels' in inputs else None,
                month_labels=inputs.get('month_labels') if use_context and 'month_labels' in inputs else None,
                state_labels=inputs.get('state_labels') if use_context and 'state_labels' in inputs else None,
                county_labels=inputs.get('county_labels') if use_context and 'county_labels' in inputs else None,
                city_labels=inputs.get('city_labels') if use_context and 'city_labels' in inputs else None,
                lat_labels=inputs.get('lat_labels'),
                lng_labels=inputs.get('lng_labels')
            )
            
            # Backward pass with gradient accumulation
            loss = loss / training_args.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log training metrics
                if global_step % training_args.logging_steps == 0:
                    writer.add_scalar('train/loss', loss.item() * training_args.gradient_accumulation_steps, global_step)
                    for loss_name, loss_value in loss_dict.items():
                        writer.add_scalar(f'train/{loss_name}_loss', loss_value.item(), global_step)
            
            # Accumulate losses
            total_loss += loss.item() * training_args.gradient_accumulation_steps
            total_climate_loss += loss_dict['climate'].item()
            total_month_loss += loss_dict['month'].item()
            total_location_loss += loss_dict['location'].item()
            total_distance_loss += loss_dict['distance'].item()
            
            # Calculate batch metrics
            batch_metrics = calculate_batch_metrics(outputs, inputs, use_context)
            
            # Accumulate metrics
            if 'correct_state' in batch_metrics:
                correct_state += batch_metrics['correct_state']
                correct_county += batch_metrics['correct_county']
                correct_city += batch_metrics['correct_city']
                total_samples += batch_metrics['total_samples']
            
            if 'total_distance_error' in batch_metrics:
                total_distance_error += batch_metrics['total_distance_error']
                if total_samples == 0:  # If we haven't counted samples from classification
                    total_samples += batch_metrics['batch_size']
        
        # Calculate average training metrics for the epoch
        num_train_batches = len(train_loader)
        train_metrics = {
            "loss": total_loss / num_train_batches,
            "climate_loss": total_climate_loss / num_train_batches,
            "month_loss": total_month_loss / num_train_batches,
            "location_loss": total_location_loss / num_train_batches,
            "distance_loss": total_distance_loss / num_train_batches,
            "avg_distance_error": total_distance_error / total_samples if total_samples > 0 else float('inf')
        }
        
        # Add classification accuracy metrics if applicable
        if use_context and total_samples > 0:
            train_metrics.update({
                "state_accuracy": correct_state / total_samples,
                "county_accuracy": correct_county / total_samples,
                "city_accuracy": correct_city / total_samples
            })
        
        # Log epoch training metrics
        for metric_name, metric_value in train_metrics.items():
            writer.add_scalar(f'train/epoch_{metric_name}', metric_value, epoch)
        
        # Evaluate on validation set (every epoch)
        logger.info(f"Evaluating on validation set at epoch {epoch + 1}")
        val_metrics = evaluate_geolocation_model(model, val_loader, device, use_context)
        
        # Log validation metrics
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f'val/{metric_name}', metric_value, epoch)
        
        # Print epoch summary
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Distance Loss: {train_metrics['distance_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Distance Loss: {val_metrics['distance_loss']:.4f}")
        if 'avg_distance_error' in val_metrics:
            logger.info(f"  Val Avg Distance Error: {val_metrics['avg_distance_error']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            model.save_pretrained(os.path.join(training_args.output_dir, 'best_model'))
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    writer.close()
    return model

def evaluate_geolocation_model(
    model: Union[CLIPModel, str],
    val_loader: DataLoader,
    device: str = 'cuda',
    use_context: bool = True
) -> Dict[str, float]:
    """Evaluate a geolocation model on a validation dataset.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to evaluate on ('cuda' or 'cpu')
        use_context: Whether the model was trained with contextual features
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Track metrics
    total_loss = 0
    total_climate_loss = 0
    total_month_loss = 0
    total_location_loss = 0
    total_distance_loss = 0
    
    # Initialize loss function
    loss_fn = GeoCLIPLoss(device)
    
    # Metrics for location accuracy
    correct_state = 0
    correct_county = 0
    correct_city = 0
    total_samples = 0
    
    # Metrics for coordinate prediction
    total_distance_error = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", ncols=80):
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(inputs['pixel_values'])
            
            # Calculate loss using the GeoCLIPLoss class
            loss, loss_dict = loss_fn(
                outputs,
                climate_labels=inputs.get('climate_labels') if use_context and 'climate_labels' in inputs else None,
                month_labels=inputs.get('month_labels') if use_context and 'month_labels' in inputs else None,
                state_labels=inputs.get('state_labels') if use_context and 'state_labels' in inputs else None,
                county_labels=inputs.get('county_labels') if use_context and 'county_labels' in inputs else None,
                city_labels=inputs.get('city_labels') if use_context and 'city_labels' in inputs else None,
                lat_labels=inputs.get('lat_labels'),
                lng_labels=inputs.get('lng_labels')
            )
            
            # Accumulate losses
            total_loss += loss.item()
            total_climate_loss += loss_dict['climate'].item()
            total_month_loss += loss_dict['month'].item()
            total_location_loss += loss_dict['location'].item()
            total_distance_loss += loss_dict['distance'].item()
            
            # Calculate batch metrics
            batch_metrics = calculate_batch_metrics(outputs, inputs, use_context)
            
            # Accumulate metrics
            if 'correct_state' in batch_metrics:
                correct_state += batch_metrics['correct_state']
                correct_county += batch_metrics['correct_county']
                correct_city += batch_metrics['correct_city']
                total_samples += batch_metrics['total_samples']
            
            if 'total_distance_error' in batch_metrics:
                total_distance_error += batch_metrics['total_distance_error']
                if total_samples == 0:  # If we haven't counted samples from classification
                    total_samples += batch_metrics['batch_size']
    
    # Calculate average metrics
    num_batches = len(val_loader)
    metrics = {
        "loss": total_loss / num_batches,
        "climate_loss": total_climate_loss / num_batches,
        "month_loss": total_month_loss / num_batches,
        "location_loss": total_location_loss / num_batches,
        "distance_loss": total_distance_loss / num_batches,
        "avg_distance_error": total_distance_error / total_samples if total_samples > 0 else float('inf')
    }
    
    # Add classification accuracy metrics if applicable
    if use_context and total_samples > 0:
        metrics.update({
            "state_accuracy": correct_state / total_samples,
            "county_accuracy": correct_county / total_samples,
            "city_accuracy": correct_city / total_samples
        })
    
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
        for step, batch in enumerate(tqdm(dataloader, desc="Profiling", ncols=80)):
            if step >= num_steps:
                break
                
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(inputs['pixel_values'])
            
            # Record step
            profiler.step()
    
    logger.info(f"Profiling complete. Results saved to {profile_output}")
    
    # Print key stats
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
