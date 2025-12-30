import torch
import torch.nn as nn

from typing import List, Tuple
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers.models.bert import BertModel
from fastNLP.modules.torch import MLP,ConditionalRandomField,allowed_transitions
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class ConvFeatureExtractionModel(nn.Module):

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        conv_dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride=1, conv_bias=False):
            padding = k // 2
            return nn.Sequential(
                nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, stride=stride, padding=padding, bias=conv_bias),
                nn.Dropout(conv_dropout),
                # nn.BatchNorm1d(n_out),
                nn.ReLU(),
                # nn.MaxPool1d(kernel_size=2, stride=2)
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for _, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(in_d, dim, k, stride=stride, conv_bias=conv_bias))
            in_d = dim

    def forward(self, x):
        # x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

batch_size = 16

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import math

class TransformerClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=4, n_layers=2, lr=0.001, max_seq_len=1024):
        super(TransformerClassifier, self).__init__()
        self.save_hyperparameters()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.position_encoding = self.create_positional_encoding(max_seq_len, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.lr = lr
        
    def create_positional_encoding(self, seq_len, hidden_dim):
        position_encoding = torch.zeros((seq_len, hidden_dim))
        for pos in range(seq_len):
            for i in range(0, hidden_dim, 2):
                position_encoding[pos, i] = math.sin(pos / (10000 ** (2 * i / hidden_dim)))
                if i + 1 < hidden_dim:
                    position_encoding[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / hidden_dim)))
        return position_encoding.unsqueeze(0)
    
    def forward(self, x, mask=None):
        
        x = self.input_linear(x)
        seq_len = x.size(1)
        x = x + self.position_encoding[:, :seq_len, :].to(x.device)

        x = x.transpose(0, 1)
        x = self.transformer(x, src_key_padding_mask=mask)
        logits = self.fc(x.transpose(0, 1))
        return logits
    
    def get_feature_embedding(self, x, mask=None):
        x = self.input_linear(x)
        seq_len = x.size(1)
        x = x + self.position_encoding[:, :seq_len, :].to(x.device)

        x = x.transpose(0, 1)
        x = self.transformer(x, src_key_padding_mask=mask)
        
        return x.transpose(0, 1)
    
    def get_linear_output(self, x, mask=None):
        x = self.input_linear(x)
        seq_len = x.size(1)
        x = x + self.position_encoding[:, :seq_len, :].to(x.device)

        x = x.transpose(0, 1)
        x = self.transformer(x, src_key_padding_mask=mask)
        out = self.fc(x.transpose(0, 1))
        return out
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        mask = (inputs[:, :, 0] == -1)
        pad_mask = (~mask).float()
        
        
        logits = self(inputs, mask=mask)
        loss = self.compute_masked_loss(logits, labels, pad_mask)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        mask = (inputs[:, :, 0] == -1)
        pad_mask = (~mask).float()
        
        logits = self(inputs, mask=mask)
        loss = self.compute_masked_loss(logits, labels, pad_mask)
        self.log('val_loss', loss)
        return loss
    
    def compute_masked_loss(self, logits, labels, mask):
        loss = self.loss_fn(logits, labels)
        
        loss = loss * mask.unsqueeze(-1)
        return loss.sum() / mask.sum()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        mask = (inputs[:, :, 0] == -1)
        logits = self(inputs, mask=mask)
        return torch.sigmoid(logits)
    
    def predict(self, data_loader):
        self.eval()
        
        all_predictions = []
        all_probas = []
        all_labels = []
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                mask = (inputs[:, :, 0] == -1)
                pad_mask = (~mask).float().unsqueeze(-1)
                logits = self(inputs, mask=mask)
                outputs = torch.sigmoid(logits)
                probas = outputs * pad_mask
                
                predicted = (outputs > 0.5).long()
                for i in range(predicted.size(0)):  # 배치 내 각 샘플에 대해
                    valid_indices = pad_mask[i].squeeze(-1).bool()  # 패딩되지 않은 위치의 인덱스
                    all_predictions.extend(predicted[i][valid_indices].cpu().numpy())
                    all_probas.extend(probas[i][valid_indices].cpu().numpy())
                    all_labels.extend(labels[i][valid_indices].cpu().numpy())

        return all_predictions, np.array(all_probas), all_labels



class MultiModalConcatLineFocalBMESBinaryClassifier(nn.Module):
    def __init__(self, id2labels, seq_len, intermediate_size = 512, num_layers=2, dropout_rate=0.1, alpha=0.5):
        super(MultiModalConcatLineFocalBMESBinaryClassifier, self).__init__()
        # feature_enc_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]
        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )
        
        self.seq_len = seq_len          # MAX Seq_len
        embedding_size = 4*64
        feature_embedding_size = 256
        combined_embedding_size = embedding_size + feature_embedding_size
        self.feature_encoder = TransformerClassifier(input_dim=376, hidden_dim=256, output_dim=1, n_heads=4, n_layers=2, lr=0.001, max_seq_len=1024)

        for param in self.feature_encoder.parameters():
            param.requires_grad = True
             
        
        self.attn = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=1)

        self.reduce_dim_layer = nn.Linear(combined_embedding_size, embedding_size)

        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=16,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                            num_layers=num_layers)

        self.position_encoding = torch.zeros((seq_len, embedding_size))
        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000**((2 * i) / embedding_size))))
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000**((2 *
                                                 (i + 1)) / embedding_size))))
        
        self.norm = nn.LayerNorm(embedding_size)
        
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf.trans_m.data *= 0
        self.binary_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        
    def focal_loss(self, inputs, targets, alpha=1, gamma=2, ignore_index=-1):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=ignore_index)
        pt = torch.exp(-ce_loss)  # pt = exp(-cross_entropy)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    # Focal Loss for binary classification
    def binary_focal_loss(self, logits, targets, mask, alpha=0.25, gamma=2):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = sigmoid(logits) if logits are provided
        focal_loss = alpha * (1 - pt) ** gamma * BCE_loss
        focal_loss = focal_loss * mask  # 마스크를 적용하여 패딩된 위치의 손실을 0으로 만듦
        return focal_loss.sum() / mask.sum()  # 유효한 위치에 대한 평균 손실 계산
    
    def compute_masked_loss(self, logits, labels, mask):
        # Binary Focal Loss 적용
        loss = self.binary_focal_loss(logits, labels, mask, alpha=0.25, gamma=2)
        return loss
        
    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    def forward(self, x, labels, ccfeature):
        mask = labels.gt(-1)
        padding_mask = ~mask
        
        x = x.transpose(1, 2)
         
        out1 = self.conv_feat_extract(x[:, 0:1, :])  
        out2 = self.conv_feat_extract(x[:, 1:2, :])  
        out3 = self.conv_feat_extract(x[:, 2:3, :])  
        out4 = self.conv_feat_extract(x[:, 3:4, :])  
        out = torch.cat((out1, out2, out3, out4), dim=2)
        position_encoding = self.position_encoding[:out.size(1), :].to(out.device)
        
        
        outputs = out + position_encoding #self.position_encoding.to(out.device)
        outputs = self.norm(outputs)
        outputs = self.encoder(outputs, src_key_padding_mask=padding_mask)
        
        with torch.no_grad():
            feature_embedding = self.feature_encoder.get_feature_embedding(ccfeature, padding_mask)
        
        if outputs.size(1) < self.seq_len:
            pad_size = self.seq_len - outputs.size(1)
            outputs = torch.nn.functional.pad(outputs, (0, 0, 0, pad_size))
        elif outputs.size(1) > self.seq_len:
            outputs = outputs[:, :self.seq_len, :]
        
        combined_out = torch.cat((outputs, feature_embedding), dim=2)  # [32, 1024, 256+256] 16]
        
        combined_out = self.reduce_dim_layer(combined_out)
        
        
        
        if combined_out.size(1) < self.seq_len:
            pad_size = self.seq_len - combined_out.size(1)
            combined_out = torch.nn.functional.pad(combined_out, (0, 0, 0, pad_size))
        elif combined_out.size(1) > self.seq_len:
            combined_out = combined_out[:, :self.seq_len, :]
        
        
        dropout_outputs = self.dropout(combined_out)
        
        logits = self.classifier(dropout_outputs)
        
        if self.training:
            # 8-class classification에서 Focal Loss 적용
            loss_class = self.focal_loss(logits.view(-1, self.label_num), labels.view(-1), alpha=1, gamma=2, ignore_index=-1)
            
            # Binary classification 부분
            logits_softmax = torch.softmax(logits, dim=-1)
            ai_probs = torch.sum(logits_softmax[:,:,4:8], dim=-1)
            binary_logits = ai_probs
            
            # Binary labels 생성: 0 for human (0-3), 1 for AI (4-7)
            binary_labels = (labels >= 4).float()
            loss_binary = self.compute_masked_loss(binary_logits, binary_labels, mask.float())
            
            loss = self.alpha * loss_class +  (1-self.alpha) * loss_binary
            output = {'loss': loss, 'logits': logits}
        else:        
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask==0] = -1
            output = {'preds': paths, 'logits': logits}
            pass

        return output
