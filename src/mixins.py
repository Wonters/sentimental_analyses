import mlflow
import torch
from tqdm import tqdm
import gc
import time
import logging

logger = logging.getLogger(__name__)


class TorchModelTrainMixin:
    """
    Mixin to use with BaseModelABC
    """

    checkpoint: str = ""
    lr: float = 2e-5
    device: torch.device

    def _train_batch(self, x, y):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        inputs.to(self.device)
        labels = y.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(**inputs)
        try:
            loss = self.criterion(outputs.logits, labels)
        except AttributeError:
            loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        logger.info(loss.item())
        mlflow.log_metric("loss", loss.item())
        mlflow.log_metric("time", time.time())
        del inputs, labels, outputs, loss
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        time.sleep(0.2)

    def train(self):
        self.init_mlflow()
        self.model.train()
        self.model.to(self.device)
        for epoch in tqdm(range(self.epoch)):
            for tweets, labels in tqdm(self.dataloader):
                try:
                    self._train_batch(tweets, labels.float())
                except RuntimeError as e:
                    logger.error(e)
                    del tweets, labels, self.optimizer
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
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
            self.scheduler.step()
        self.save()
        super().train()


class SklearnModelTrainMixin:
    """
    Mixin to use with BaseModelABC
    """

    def train(self):
        self.model.fit(self.x_train, self.y_train)
        super().train()
