import mlflow
import torch
from tqdm import tqdm
import numpy as np
import gc
import time
import logging
import optuna
import random
from torch.utils.data import Subset, DataLoader, DistributedSampler
import torch.distributed as dist
import pandas as pd

logger = logging.getLogger(__name__)


class TorchModelTrainMixin:
    """
    Mixin to use with BaseModelABC
    """

    checkpoint: str = ""
    lr: float = 2e-5
    device: torch.device

    def sample_dataset(self, frac=0.1):
        dataset_size = len(self.dataset)
        sample_size = int(frac * dataset_size)
        indices = random.sample(range(dataset_size), sample_size)
        sampled_dataset = Subset(self.dataset, indices)
        return sampled_dataset

    def get_sampled_dataloader(self, frac=0.1):
        sampled_dataset = self.sample_dataset(frac=frac)
        return DataLoader(sampled_dataset, batch_size=self.batch_size, shuffle=True)

    def optuna_train(self, run_name:str = "", n_trials:int=30, frac=0.1):
        self.init_mlflow(run_name)
        self.dataloader = self.get_sampled_dataloader(frac=frac)
        self.x_val = self.x_val.sample(frac=frac, random_state=42)
        self.y_val = self.y_val.sample(frac=frac, random_state=42)
        self.x_train = self.x_train.sample(frac=frac, random_state=42)
        self.y_train = self.y_train.sample(frac=frac, random_state=42)
        study = optuna.create_study(direction="maximize",
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
        study.optimize(self.objective, n_trials=n_trials)
    
    def params_optim(self, trial):
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
        gamma = trial.suggest_float('gamma', 0.1, 0.9)
        step_size = trial.suggest_int('step_size', 2, 10)
        return {'lr': lr, 'gamma': gamma, 'step_size': step_size} 

    def objective(self, trial):
        kwargs = self.params_optim(trial)
        with mlflow.start_run(nested=True):
            mlflow.log_params(kwargs)
            self.reinit_scheduler_optimizer(**kwargs)
            acc = self.train()
        return acc
    
    def get_ddp_dataloader(self, frac=1.0):
        sampled_dataset = self.sample_dataset(frac=frac)
        sampler = DistributedSampler(sampled_dataset)
        dataloader = DataLoader(sampled_dataset, batch_size=self.batch_size, sampler=sampler)
        return dataloader, sampler
    

    def _train_batch(self, x, y):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        if isinstance(inputs, dict) and inputs["input_ids"]:
            inputs["input_ids"] = inputs["input_ids"].float()
        # todo : fix this case for bert vs lstm
        if True and isinstance(y, torch.Tensor) and y.dtype == torch.float32:
            labels = y.long()
        else:
            labels = y.float()
        self.optimizer.zero_grad()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        # Give input_ids and attention masks
        outputs = self.model(**inputs)
        try:
            loss = self.criterion(outputs.logits, labels)
            _, preds = torch.max(outputs.logits, dim=1)
        except AttributeError:
            loss = self.criterion(outputs, labels)
            # todo : error on accuracy return always 0 
            preds = outputs 
        correct = (preds == labels).sum().item()    
        acc = correct / len(labels)
        try:
            loss.backward()
        except Exception as e:
            if "out of memory" in str(e):
                    logger.error(f"[Rank {dist.get_rank()}] CUDA OOM detected")
                    torch.cuda.empty_cache()
                    gc.collect()
                    return 0.0  # skip the batch
            else:
                raise e
        self.optimizer.step()
        logger.info(f" Rank {dist.get_rank()} loss {loss.item()} acc {acc}")
        mlflow.log_metric("loss", loss.item())
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("time", time.time())
        del inputs, labels, outputs, loss
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        time.sleep(0.2)
        return acc

    def train(self):
        try:
            self.init_mlflow()
        except Exception:
            logger.info("mlflow run already started, you had launched train with optuna")
            pass
        self.model.train()
        self.model.to(self.device)
        acc = []
        for epoch in tqdm(range(self.epoch)):
            for tweets, labels in tqdm(self.dataloader):
                try:
                    current_acc = self._train_batch(tweets, labels.float())
                    acc.append(current_acc)
                except RuntimeError as e:
                    logger.error(e)
                    del tweets, labels, self.optimizer
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    time.sleep(0.2)
                    self.save()
                    self.load_checkpoint()
                    self.model.to(self.device)
                    self.model.train()
                    continue
                if torch.backends.mps.is_available():
                    logger.info(
                        f"MPS allocated memory: {torch.mps.driver_allocated_memory()}"
                    )
                if torch.cuda.is_available():
                    logger.info(
                        f"CUDA allocated memory: {torch.cuda.memory_allocated()}"
                    )
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(current_acc)
            else:   
                self.scheduler.step()
        super().train()
        return sum(acc)/len(acc)


class SklearnModelTrainMixin:
    """
    Mixin to use with BaseModelABC
    """

    def train(self):
        self.model.fit(self.x_train, self.y_train)
        super().train()
