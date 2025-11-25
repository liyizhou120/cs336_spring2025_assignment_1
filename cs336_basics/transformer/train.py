import torch 
import numpy as np 
import numpy.typing as npt 
import wandb 
from tqdm import tqdm 
from loguru import logger 

import cs336_basics.transformer.trainer_util as utils
import cs336_basics.transformer.module as module 

if __name__ == "__main__":
    logger.add("./data/log/train_v0.log", rotation="1 day", retention = "7 days", level = "INFO")
    
    # Setting hyperparameters 
    # Model parameters
    model_config = {
        "vocab_size": 10000,      
        "context_length": 256, 
        "num_layers": 4, 
        "num_heads":16,
        "d_model": 512, 
        "d_ff": 1344, 
        "rope_theta": 10000, 
    }

    #Optimizer Config 
    optim_config = {
        "lr": 3e-4,                 # Learning rate
        "weight_decay": 1e-2,       # Weight decay 
        "betas": (0.9, 0.999),      # AdamW beta 
        "max_norm": 1.0, 
    }
    
    # Training parameters 
    train_config = {
        "batch_size": 16,           # Batch size 
        "total_epochs": 0.5,        # Training epoch
        "checkpoint_freq": 2000,    # Checkpoint after n steps 
        "log_freq": 10,             # log after n steps 
        "val_freq": 400,            # evaluation after n steps
        "val_batch_size": 16,       # batch size when evaluating 
        "val_batches": 20,          # batch numbers when evaluating 
    }
    
    # Data paths
    
    data_paths = {
        "training_dataset_path": "/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/token/TinyStories_train_10000_token_ids.npy",
        "validation_dataset_path": "/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/token/TinyStories_valid_10000_token_ids.npy",  # 验证集路径
        "checkpoint_load_path": None,  # 模型检查点路径
        "checkpoint_save_format": "/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/model/checkpoint_v0_{}.pt",  # 检查点保存路径格式
        "final_model_path": "/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/model/final_model_v0.pt",  # 最终模型保存路径
    }
    
    # Initialization of wandb 
    run = wandb.init(
        project="cs336-assignment-1",
        name = "train_v1",
        config={
            "model": model_config, 
            "optimizer": optim_config, 
            "training": train_config,
        }
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info("Start initializing model")
    model = module.TransformerLM(
        vocab_size=model_config["vocab_size"], 
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_model = model_config["d_model"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
        device = device,
    )
    logger.info("Initialization completed")
    
    # Initializing optimizer 
    logger.info("Start initializing optimizer")
    optimizer = utils.AdamWOptimizer(
        model.parameters(), 
        lr = optim_config["lr"],
        weight_decay=optim_config["weight_decay"],
        betas = optim_config["betas"],
    )
    logger.info("Optimizer initialization completed")
    
    #If there is checkpoint, then load it 
    start_iter = 1 
    if data_paths["checkpoint_load_path"]:
        logger.info(f"Start loading checkpoint:{data_paths['checkpoint_load_path']}")
        start_iter = utils.load_checkpoint(
            data_paths["checkpoint_load_path"],
            model = model, 
            optimizer=optimizer,
        )
        start_iter += 1
        logger.info(f"Model checkpoint has been loaded successfully. Current iteration is : {start_iter}")
    else: 
        logger.info("Not providing checkpoint. Begin from start")
    
    logger.info(f"Start loading training dataset: {data_paths['training_dataset_path']}, validation set: {data_paths['validation_dataset_path']}")
    training_dataset = np.load(data_paths['training_dataset_path'], mmap_mode='r+')  #Memory Mapping 
    validation_dataset = None
    if data_paths['validation_dataset_path']: 
        validation_dataset = np.load(data_paths['validation_dataset_path'], mmap_mode = 'r+')
    logger.info("Dataset loaded successfully")
    
    # Compute training steps 
    total_tokens = training_dataset.shape[0]
    total_steps = int(train_config["total_epochs"] * total_tokens) // (train_config["batch_size"] * model_config["context_length"])
    logger.info(f"总token数: {total_tokens}, 训练轮数: {train_config['total_epochs']}, batch大小: {train_config['batch_size']}, 上下文长度: {model_config['context_length']}")
    logger.info(f"总训练步数: {total_steps}")
    
    # Step looping 
    logger.info("Start training models...")
    for step in tqdm(range(start_iter, total_steps + 1), desc = "Training progress", unit="step"):
        optimizer.zero_grad()
        
        # Cosine schedule
        lr_now = utils.learning_rate_cosine_schedule(
            it = step, 
            max_learning_rate= optim_config["lr"],
            min_learning_rate= optim_config["lr"] * 0.01,
            warmup_iters=int(0.05 * total_steps), 
            cosine_cycle_iters = total_steps,   
        )
        for param_group in optimizer.param_groups: 
            param_group['lr'] = lr_now
        
        # Getting batch data 
        inputs, targets = utils.get_batch(
            training_dataset, 
            batch_size = train_config["batch_size"], 
            context_length=model_config["context_length"],
            device = device,
        )
        
        # Forward Pass 
        logits = model(inputs)
        
        # Compute losses 
        loss = utils.cross_entropy(logits, targets)
        
        #Backward prop and optimization 
        loss.backward() 
        
        # Computing L2 Norm for gradient 
        if step % train_config["log_freq"] == 0: 
            grad_norm = utils.compute_grad_norm(model.parameters())
        
        utils.gradient_clipping(model.parameters(), max_norm=optim_config["max_norm"])  # gradient clipping 
        optimizer.step()
        
        
        if step % train_config["log_freq"] == 0: 
            logger.info(f"Step{step}, Loss: {loss.item()}, Grad L2 Norm: {grad_norm}")
            
            # Logging loss and gradient norm using wandb 
            wandb.log({
                "training loss": loss.item(), 
                "lr": lr_now, 
                "grad_l2_norm": grad_norm, 
                "step": step
            })
        
        # Evaluation model on validation dataset
        if validation_dataset is not None and step % train_config["val_freq"] == 0: 
            logger.info(f"Evaluating on validation dataset")
            val_loss = utils.evaluate_model(
                model=model,
                dataset=validation_dataset, 
                device=device, 
                batch_size = train_config["val_batch_size"], 
                context_length=model_config["context_length"], 
                num_batches=train_config["val_batches"]
            )  
            logger.info(f"validation loss: {val_loss}")
            wandb.log({"val_loss": val_loss, "step": step})
        
        # Saving checkpoint 
        if step % train_config["checkpoint_freq"] == 0: 
            checkpoint_save_path = data_paths["checkpoint_save_format"].format(step)
            logger.info(f"Saving model checkpoint to: {checkpoint_save_path}")
            utils.save_checkpoint(
                model=model,
                optimizer=optimizer, 
                iteration = step, 
                out=checkpoint_save_path
            )
            logger.info("Model checkpoint saved successfully")
    
    
    logger.info("Model training completed")
        
    # Save the final model 
    logger.info(f"Saving the final model to: {data_paths['final_model_path']}")
    
    utils.save_checkpoint(
        model=model, 
        optimizer=optimizer, 
        iteration=total_steps, 
        out = data_paths["final_model_path"],
    )    
    
    logger.info("Final model saved")
    
    # Close wandb 
    wandb.finish()
            
        
        