import torch
from torch.optim import Adam
from model.modules.ggnn import GGNN
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlastDectClassifier(pl.LightningModule):
    def __init__(self,lr=1e-3, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()
        #### Graph encoder (GGNN, GATv2, GCN)
        self.graph_model = GGNN(input_dim=100, output_dim=128, max_edge_types=1, read_out="mean")
        #### loss function
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
        #### Optimizer params
        self.lr=lr
        self.weight_decay=weight_decay
    

    def forward(self, x):
        x = self.graph_model(x)

        return x

    def training_step(self, batch, batch_idx):
        g, labels = batch
        logits = self.graph_model(g)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        loss = self.loss_func(logits, labels)
        preds = torch.argmax(logits, dim=1)

        return loss
    
    def validation_step(self, batch, batch_idx):
        g, labels = batch
        logits = self.graph_model(g)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        loss = self.loss_func(logits, labels)
        preds = torch.argmax(logits, dim=1)

        return loss
    
    def test_step(self, batch, batch_idx):
        g, labels = batch
        logits = self.graph_model(g)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        loss = self.loss_func(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss
    
    def on_train_epoch_end(self):
        # Compute metrics
        pass

        
    def on_validation_epoch_end(self):
        pass
        
        

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)
        return optimizer


