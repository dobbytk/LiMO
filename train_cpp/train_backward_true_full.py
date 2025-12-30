import os
import sys
import json
import torch
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn
import random
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional
import numpy as np
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from numpyencoder import NumpyEncoder
import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from tqdm import tqdm, trange
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import xml.etree.ElementTree as ET
warnings.filterwarnings('ignore')

project_path = os.path.abspath('')
if project_path not in sys.path:
    sys.path.append(project_path)
#import backend_model_info

from dataloader import DataManager, DataManagerTest
from model_4 import MultiModalConcatLineFocalBMESBinaryClassifier

from sklearn.metrics import roc_curve, precision_recall_curve, auc, classification_report

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
from collections import Counter, defaultdict

with open("/KSY_copy/train_cpp/cpp_errorcode.txt") as f:
    error_codes = [error.replace('\n', '') for error in f.readlines()]
    
def analyze_cpplint_output(eval_result: str, error_codes: list) -> Counter:
    analysis = [0]*len(error_codes)
    
    error_pattern = re.compile(r'\[([a-z0-9_]+\/[a-z0-9_]+)\]')
    errors = error_pattern.findall(eval_result)

    
    error_counts = Counter(errors)
    
    analysis = [error_counts[e] for e in error_codes]
            
    return analysis


def analyze_cpplint_outputs(eval_results: list, error_codes: list) -> Counter:
    analysis = {}
    for e in error_codes:
        analysis[e] = [0]*len(eval_results)
    
    error_pattern = re.compile(r":(\d+):.*?\[([a-z]+[\/-][a-z0-9_-]+)\]")
    for idx, eval_result in enumerate(eval_results):
        errors = error_pattern.findall(eval_result)
    
        
        error_counts = Counter(errors)
        for error_code, count in error_counts.items():
            analysis[error_code][idx] = count
            
            
    return analysis


def analyze_cpplint_output_line(eval_result: str, total_lines: int, error_codes: list):
    
    error_pattern = re.compile(r":(\d+):.*?\[([a-z]+[\/-][a-z0-9_-]+)\]")
    errors = error_pattern.findall(eval_result)
    
    
    line_error_counts = defaultdict(Counter)
    
    for line, code in errors:
        line_error_counts[int(line)][code] += 1

    
    analysis = [[0]*len(error_codes) for _ in range(total_lines)]
    
    
    for line in range(total_lines):
        if line in line_error_counts:
            analysis[line] = [line_error_counts[line][code] for code in error_codes]
    
    return analysis


def split_code_sentence(code, use_sp=False):
        import re
        pattern = re.compile(
        r'"""|\'\'\'|"|\'|#|==|'
        r'\n|'
        r'[^\S\n]+|'
        r'\w+|[.,()\[\]{};:\=\_\+\-\*\/\~\!\%\^\&\<\>\?]')
        
        tokens = pattern.findall(code)
        return tokens

def ccfeature_line_to_token_level(code):
    code_tokens = split_code_sentence(code)
    count = 0
    line_num_list = []
    for token in code_tokens:
        line_num_list.append(count)
        if token == '\n':
            count += 1
    return line_num_list[:1024]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            if obj.dim() == 0:
                return obj.item()
            else:
                return obj.tolist()
        return super(NpEncoder, self).default(obj)

class CustomDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = [int(idx) for idx in indices]

    def __getitem__(self, index):
        real_idx = self.indices[index]
        return self.original_dataset[int(real_idx)]

    def __len__(self):
        return len(self.indices)
    

def get_roc_metrics(true_labels, pred_labels):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)
    roc_auc = auc(fpr, tpr)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr[ix], 1-fpr[ix], J[ix]))
    return float(roc_auc)

class SupervisedTrainer:
    def __init__(self, data, model, en_labels, id2label, args):
        self.data = data
        self.model = model
        self.en_labels = en_labels
        self.id2label = id2label

        self.seq_len = args.seq_len
        self.num_train_epochs = args.num_train_epochs
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.warm_up_ratio = args.warm_up_ratio

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self._create_optimizer_and_scheduler()
        
        self.best_val_loss = float('inf')
        self.best_f1_score = 0.0
        self.best_model_path = None
        self.writer = None
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-1)
        self.threshold = 0.5

    def _create_optimizer_and_scheduler(self):
        num_training_steps = len(
            self.data.train_dataloader) * self.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]

        named_parameters = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.weight_decay,
            },
            {
                "params": [
                    p for n, p in named_parameters
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.warm_up_ratio * num_training_steps),
            num_training_steps=num_training_steps)

    def train(self, ckpt_name='linear_en.pt', prediction_method="most_common"):
        
        for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_steps = 0
            # train
            for step, inputs in enumerate(
                    tqdm(self.data.train_dataloader, desc="Iteration")):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.set_grad_enabled(True):
                    labels = inputs['labels']
                    output = self.model(inputs['features'], inputs['labels'], inputs['ccfeatures'])#, inputs['line_indices'])
                    logits = output['logits']
                    loss = output['loss']
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                            
                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_steps += 1
            
                if step % 50 == 0:
                    self.writer.add_scalar('Training Loss', loss.item(), epoch * len(self.data.train_dataloader) + step)
            
            
            avg_train_loss = tr_loss / nb_tr_steps
            print(f'epoch {epoch+1}: train_loss {avg_train_loss}')
            self.writer.add_scalar('Average Training Loss', avg_train_loss, epoch)
            
            val_loss, sent_result = self.valid(prediction_method=prediction_method)
            self.writer.add_scalar('Validation Loss', val_loss, epoch)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = f"{ckpt_name}"
                self.writer.add_scalar('Best Validation Loss', self.best_val_loss, epoch)
                torch.save(self.model.cpu(), self.best_model_path)
                self.model.to(self.device)
        
        if self.best_model_path:
            print(f"Reloading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path).state_dict())
            self.model.to(self.device)
        
        self.writer.close()
        return
    
    def valid(self, content_level_eval=False, prediction_method="most_common"):
        self.model.eval()
        texts = []
        true_labels = []
        pred_labels = []
        total_logits = []
        total_probs = []
        total_loss = 0.0
        total_steps = 0
        
        for step, inputs in enumerate(
                tqdm(self.data.val_dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                labels_ = inputs['labels']
                output = self.model(inputs['features'], inputs['labels'], inputs['ccfeatures'])
                preds = output['preds']
    
                logits_ = output['logits']
                
                probabilities = F.softmax(logits_, dim=-1)
                
                logits = logits_.view(-1, logits_.size(-1))
                labels = labels_.view(-1)
                loss = self.loss_function(logits, labels)
                total_loss += loss.item()
                total_steps += 1

                texts.extend(inputs['text'])
                pred_labels.extend(preds.cpu().tolist())
                true_labels.extend(labels_.cpu().tolist())
                total_probs.extend(probabilities)

        avg_val_loss = total_loss / total_steps
        print(f"Validation Loss: {avg_val_loss}")
        
        print("*" * 8, "Sentence Level Evalation", "*" * 8)
        
        sent_result = self.sent_level_eval(texts, true_labels, pred_labels, total_probs, prediction_method)
        
        return avg_val_loss, sent_result
    
    def test(self, test_dataloader, content_level_eval=False, prediction_method="most_common"):
        self.model.eval()
        texts = []
        true_labels = []
        pred_labels = []
        total_logits = []
        total_probs = []
        problem_ids = []
        user_ids = []
        
        for step, inputs in enumerate(
                tqdm(test_dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                labels = inputs['labels']
                output = self.model(inputs['features'], inputs['labels'], inputs['ccfeatures'])#, inputs['line_indices'])
                logits = output['logits']
                preds = output['preds']
                # problem_id = inputs['problem_id']
                # user_id = inputs['user_id']
                
                probabilities = F.softmax(logits, dim=-1)

                texts.extend(inputs['text'])
                pred_labels.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                # problem_ids.extend(problem_id)
                # user_ids.extend(user_id)
                total_logits.extend(logits.cpu().tolist())
                total_probs.extend(probabilities)
        
        line_counts = [len(text.split('\n')) for text in texts]
        
        if content_level_eval:
            # content level evaluation
            print("*" * 8, "Content Level Evalation", "*" * 8)
            content_result = self.content_level_eval(texts, true_labels, pred_labels, total_probs, prediction_method)
        else:
            content_result = None
        print("*" * 8, "Sentence Level Evalation", "*" * 8)
        #word_result, sent_result = self.sent_level_eval(texts, true_labels, pred_labels, total_probs, prediction_method)
        sent_result = self.sent_level_eval(texts, true_labels, pred_labels, total_probs, prediction_method)
            
        # return sent_result, content_result, {'text':texts,'pred':pred_labels, 'true':true_labels, 'problem_id':problem_ids, 'user_id': user_ids}
        return sent_result, content_result, {'text': texts, 'pred': pred_labels, 'true': true_labels, 'line_count':line_counts, 'probs':total_probs}

    
    def content_level_eval(self, texts, true_labels, pred_labels, pred_probs, prediction_method='most_common'):
        if prediction_method =='threshold':
            threshold = self.threshold
        else:
            threshold = None
            pred_labels_threshold = pred_labels
        
        true_content_labels = []
        pred_content_labels = []
        pred_content_probs = []
        
        for text, true_label, pred_label, pred_prob in zip(texts, true_labels, pred_labels_threshold, pred_probs):
            true_label = np.array(true_label)
            pred_label = np.array(pred_label)
            pred_prob = np.array(pred_prob)
            
            mask = true_label != -1
            true_label = true_label[mask].tolist()
            pred_label = pred_label[mask].tolist()
            
            pred_prob = torch.tensor(pred_prob[mask])
            true_common_tag = self._get_most_common_tag(true_label)
            true_content_labels.append(true_common_tag[0])
            
            pred_common_tag = self._get_most_common_tag(pred_label)
            pred_content_labels.append(pred_common_tag[0])
            
            cont_prob = pred_prob[:, 4:8].sum(dim=1)
            pred_content_prob = torch.mean(cont_prob, dim=0)
            pred_content_probs.append(pred_content_prob.item())
            
        true_content_labels = [self.en_labels[label] for label in true_content_labels]
        pred_content_labels = [self.en_labels[label] for label in pred_content_labels]
        
        result = self._get_precision_recall_acc_f1(true_content_labels, pred_content_labels, pred_content_probs)
        
        return result

    def sent_level_eval(self, texts, true_labels, pred_labels, pred_probs, prediction_method='most_common'):
        if prediction_method =='threshold':
            threshold = self.threshold
        else:
            threshold = None
            pred_labels_threshold = pred_labels
        
        # For line-wise labeling
        true_sent_labels = []
        pred_sent_labels = []
        pred_sent_probs = []
        for text, true_label, pred_label, pred_prob in zip(texts, true_labels, pred_labels_threshold, pred_probs):
            true_label = np.array(true_label)
            pred_label = np.array(pred_label)
            pred_prob = np.array(pred_prob.cpu())
            mask = true_label != -1
            true_label = true_label[mask].tolist()
            pred_label = pred_label[mask].tolist()
            pred_prob = torch.tensor(pred_prob[mask])
            sents = text.split('\n')
            for true_label_idx in range(len(true_label)):
                if sents[true_label_idx] == '' or sents[true_label_idx].isspace():  # 빈 문장일 경우 처리하지 않음
                    continue
                true_sent_label = self.id2label[true_label[true_label_idx]]
                pred_sent_label = self.id2label[pred_label[true_label_idx]]
                
                true_sent_labels.append(true_sent_label.split('-')[-1])
                pred_sent_prob = pred_prob[true_label_idx, 4:8].sum()
                pred_sent_probs.append(pred_sent_prob.item())
                pred_sent_labels.append(pred_sent_label.split('-')[-1])
            
        true_sent_labels = [self.en_labels[label] for label in true_sent_labels]
        pred_sent_labels = [self.en_labels[label] for label in pred_sent_labels]
        
        sent_result = self._get_precision_recall_acc_f1(true_sent_labels, pred_sent_labels, pred_sent_probs)
        return sent_result
    
    
    def _get_threshold_tag(self, logits, machine_threshold=0.5):
        human_logits = logits[:, :, :4]  # Human Classes
        machine_logits = logits[:, :, 4:] # Machine Classes
        human_scores = torch.sum(human_logits, dim=-1)  # Shape: [batch_size, seq_len]
        machine_scores = torch.sum(machine_logits, dim=-1)        # Shape: [batch_size, seq_len]
        pred_labels = torch.where(machine_scores >= machine_threshold, 4, 0)  # 0 for Human, 4 for AI
        
        return pred_labels.cpu().tolist()
    
    def _get_most_common_tag(self, tags):
        """most_common_tag is a tuple: (tag, times)"""
        from collections import Counter
        tags = [self.id2label[tag] for tag in tags]
        tags = [tag.split('-')[-1] for tag in tags]
        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0]
        return most_common_tag
    
    def _get_precision_recall_acc_f1(self, true_labels, pred_labels, pred_probs=None, pos_label: int = 1) -> Dict[str, Any]:
        """
        true_labels: [0/1]
        pred_labels: 이미 threshold가 적용된 0/1 예측
        pred_probs : 선택. 점수(양성=pos_label의 확률/로짓 등). 있으면 ROC/AUPRC과 임계값 탐색 리포트 추가.
        pos_label  : 양성 클래스(기본 1)
        """
        y_true = np.asarray(true_labels).astype(int)
        y_pred = np.asarray(pred_labels).astype(int)

        
        acc  = accuracy_score(y_true, y_pred)
        mF1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
        bF1  = f1_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
        prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
        cm   = confusion_matrix(y_true, y_pred, labels=[0,1])

        print("=== Given labels (as-is) ===")
        print("Accuracy: {:.3f}".format(acc*100))
        print("Macro F1 Score: {:.3f}".format(mF1*100))
        print("Binary F1 Score (pos): {:.3f}".format(bF1*100))
        print("Precision/Recall per class:")
        print("{:.1f},{:.1f},{:.1f},{:.1f}".format(prec[0]*100, rec[0]*100, prec[1]*100, rec[1]*100))
        print(f"CM [[TN FP],[FN TP]] = {cm.tolist()}")

        
        result: Dict[str, Any] = {
            "given_labels": {
                "accuracy": acc, "macro_f1": mF1, "binary_f1": bF1,
                "precision": prec, "recall": rec, "cm": cm
            },
            "roc_auc": None,
            "auprc": None,
            "thresholds": {}
        }

        
        if pred_probs is not None:
            y_score = np.asarray(pred_probs, dtype=float)

            # ROC / AUPRC
            try:
                fpr, tpr, thr_roc = roc_curve(y_true, y_score, pos_label=pos_label)
                roc_auc = float(auc(fpr, tpr))
            except Exception:
                roc_auc = None

            try:
                auprc = float(average_precision_score(y_true, y_score, pos_label=pos_label))
            except Exception:
                auprc = None
            try:
                prec_curve, rec_curve, thr_pr = precision_recall_curve(y_true, y_score, pos_label=pos_label)
                result["pr_curve"] = {
                    "precision": prec_curve.tolist(),
                    "recall": rec_curve.tolist(),
                    "thresholds": thr_pr.tolist()
                }
            except Exception:
                result["pr_curve"] = None
            print(f"ROC_AUC (fpr-tpr): {roc_auc:.3f}" if roc_auc is not None else "ROC_AUC: N/A")
            print(f"AUPRC: {auprc:.3f}" if auprc is not None else "AUPRC: N/A")

            # Helper: 특정 threshold에서 평가
            def eval_at(thr: float, tag: str) -> Dict[str, Any]:
                y_hat = (y_score > thr).astype(int)
                acc_  = accuracy_score(y_true, y_hat)
                mF1_  = f1_score(y_true, y_hat, average='macro', zero_division=0)
                bF1_  = f1_score(y_true, y_hat, average='binary', pos_label=pos_label, zero_division=0)
                pr_   = precision_score(y_true, y_hat, average=None, zero_division=0)
                rc_   = recall_score(y_true, y_hat, average=None, zero_division=0)
                cm_   = confusion_matrix(y_true, y_hat, labels=[0,1])
                print(f"[{tag}] thr={thr:.3f} | Acc={acc_*100:.1f}  MacroF1={mF1_*100:.1f}  BinF1(pos)={bF1_*100:.1f}")
                print(" P/R per class -> 0(H): {:.1f}/{:.1f} , 1(AI): {:.1f}/{:.1f}".format(pr_[0]*100, rc_[0]*100, pr_[1]*100, rc_[1]*100))
                print(f" CM [[TN FP],[FN TP]] = {cm_.tolist()}")
                return {"thr": float(thr), "accuracy": acc_, "macro_f1": mF1_, "binary_f1": bF1_, "precision": pr_, "recall": rc_, "cm": cm_}

            # Youden J (TPR - FPR) 최대
            def best_thr_youden() -> float:
                if roc_auc is None or len(thr_roc) == 0:
                    return 0.5
                J = tpr - fpr
                i = int(np.argmax(J))
                return float(thr_roc[i])

            # 양성 F1 최대(PR 기반)
            def best_thr_posF1() -> float:
                prec_curve, rec_curve, thr_pr = precision_recall_curve(y_true, y_score, pos_label=pos_label)
                if len(thr_pr) == 0:
                    return 0.5
                f1_curve = (2 * prec_curve * rec_curve) / (prec_curve + rec_curve + 1e-12)
                i = int(np.nanargmax(f1_curve[:-1]))  # 마지막 점은 threshold 없음
                return float(thr_pr[i])

            thr05     = 0.5
            thrJ      = best_thr_youden()
            thrBestF1 = best_thr_posF1()

            print("=== Threshold sweeps on scores ===")
            res05  = eval_at(thr05, "thr=0.5")
            resJ   = eval_at(thrJ, "thr=YoudenJ")
            resF1  = eval_at(thrBestF1, "thr=bestPosF1")

            result.update({
                "roc_auc": roc_auc,
                "auprc": auprc,
                "thresholds": {
                    "thr@0.5": res05,
                    "thr@youden": resJ,
                    "thr@best_posF1": resF1
                }
            })
        else:
            print("ROC_AUC (fpr-tpr): N/A (pred_probs is None)")
            print("AUPRC: N/A (pred_probs is None)")

        # CSV 한 줄 요약(기존 포맷과 유사)
        pr_line = "{:.1f},{:.1f},{:.1f},{:.1f}".format(prec[0]*100, rec[0]*100, prec[1]*100, rec[1]*100)
        print("{:.1f},{:.1f},{:.1f},{},{:.3f},{}".format(
            acc*100, mF1*100, bF1*100, pr_line, result["roc_auc"] if result["roc_auc"] is not None else float("nan"),
            f"{result['auprc']:.3f}" if result["auprc"] is not None else "N/A"
        ))

        return result



def construct_bmes_labels(labels):
    prefix = ['B-', 'M-', 'E-', 'S-']
    id2label = {}
    counter = 0

    for label, id in labels.items():
        for pre in prefix:
            id2label[counter] = pre + label
            counter += 1
    
    return id2label

def remove_duplicates(prob_dict):
    total_p = 0
    total = 0
    for problem_id, entries in prob_dict.items():
        n = 0
        unique_texts = set()
        unique_entries = []
        
        for entry in entries:
            if entry['text'] not in unique_texts:
                unique_entries.append(entry)
                unique_texts.add(entry['text'])
            else:
                n += 1
        if n != 0:
            total_p += 1
        total += n
        
        prob_dict[problem_id] = unique_entries     

from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from collections import Counter

def warn_group_overlap(groups_arr, idx_a, idx_b, name_a="A", name_b="B"):
    ga = set(groups_arr[idx_a])
    gb = set(groups_arr[idx_b])
    inter = ga & gb
    if inter:
        print(f"[WARN] {name_a} and {name_b} share {len(inter)} problem_ids (leak risk).")
    else:
        print(f"[OK] No problem_id overlap between {name_a} and {name_b}.")

def split_dataset(data_path, dataset, seed=42, test_size=0.2, val_size=0.1):
    
    with open(os.path.join(data_path, f"codenet(cpp)_{dataset}_features.jsonl"), "r", encoding="utf-8") as f:
        full_train_set = [json.loads(line) for line in f]
    
    seed_everything(seed)

    
    for i, sample in enumerate(full_train_set):
        
        if sample.get("problem_id") is None:
            sample["problem_id"] = f"__none__#{i}"

        if 'line' in dataset:
            n_lines = len(sample.get('text', '').split('\n'))
            ccfeature_line = analyze_cpplint_output_line(sample.get('eval', ''), n_lines, error_codes)
            
            sample['ccfeature'] = ccfeature_line
        else:
            sample['ccfeature'] = analyze_cpplint_output(sample.get('eval', ''), error_codes)


    labels = np.array([sample['label'] for sample in full_train_set])
    groups = np.array([sample['problem_id'] for sample in full_train_set])


    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_full_idx, test_idx = next(gss.split(np.zeros(len(full_train_set)), labels, groups=groups))


    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(
        gss_val.split(
            np.zeros(len(train_full_idx)),
            labels[train_full_idx],
            groups=groups[train_full_idx]
        )
    )

    train_idx = train_full_idx[train_idx]
    val_idx   = train_full_idx[val_idx]


    warn_group_overlap(groups, train_idx, val_idx, "Train", "Val")
    warn_group_overlap(groups, train_idx, test_idx, "Train", "Test")
    warn_group_overlap(groups, val_idx,   test_idx, "Val",   "Test")


    train_set = [full_train_set[i] for i in train_idx]
    val_set   = [full_train_set[i] for i in val_idx]
    test_set  = [full_train_set[i] for i in test_idx]


    def distrib(name, arr):
        c = Counter([s['label'] for s in arr])
        total = len(arr)
        print(f"{name}: {total}  | human={c.get('human',0)} ({c.get('human',0)/total:.2%}), AI={c.get('AI',0)} ({c.get('AI',0)/total:.2%})")

    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")
    distrib("Train", train_set)
    distrib("Val",   val_set)
    distrib("Test",  test_set)
    
    

    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")
    
    
    return [train_set, val_set, test_set]



import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Transformer')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train_mode', type=str, default='classify')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--method', type=str, default="focalbmesbinary_embedconcat_transformer256")
    
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_dataset', action='store_true')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--valid_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')

    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)

    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_content', action='store_true')
    
    parser.add_argument('--ckpt_name', type=str, default='')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--testbed', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("Log INFO: split dataset...")
    df_ = split_dataset(data_path=args.data_path, dataset=args.dataset, seed=args.seed)  # [train, val, test]

    en_labels = {
        'human': 0,
        'AI': 1
    }
    
    id2label = construct_bmes_labels(en_labels)
    label2id = {v: k for k, v in id2label.items()}

    prediction_method = 'most_common'

    experiment_results = []

    if 'revised' in args.dataset:
        datas = DataManagerTest(datas=df_, batch_size=args.batch_size, max_len=args.seq_len, human_label='human', id2label=id2label)
    else:
        datas = DataManager(datas=df_, batch_size=args.batch_size, max_len=args.seq_len, human_label='human', id2label=id2label)

    # classifier 선택
    if args.method == 'focalbmesbinary_embedconcat_transformer256':
        if args.testbed == 'toplevel':
            if 'gemini' in args.dataset or 'gpt4' in args.dataset:
                classifier = MultiModalConcatLineFocalBMESBinaryClassifier(id2labels=id2label, seq_len=args.seq_len, alpha=args.alpha)

    ckpt_name = f'ckpt/cpp/{args.ckpt_name}_best_f1.pt'

    trainer = SupervisedTrainer(datas, classifier, en_labels, id2label, args)
    trainer.writer = SummaryWriter(log_dir=f"runs/cpp/{args.ckpt_name}")

    experiment_result = {}

    if args.do_test:
        print("Log INFO: do test...")
        saved_model = torch.load(ckpt_name)
        trainer.model.load_state_dict(saved_model.state_dict())
        if 'hybrid' in args.dataset or 'revised' in args.dataset:
            test_sent_result, _, test_raw_results = trainer.test(datas.test_dataloader, content_level_eval=False, prediction_method=prediction_method)
            experiment_result['test_result'] = {'line': test_sent_result, 'raw': test_raw_results}
        else:
            test_sent_result, test_content_result, test_raw_results = trainer.test(datas.test_dataloader, content_level_eval=True, prediction_method=prediction_method)
            experiment_result['test_result'] = {'line': test_sent_result, 'document': test_content_result, 'raw': test_raw_results}
    else:
        print("Log INFO: do train...")
        trainer.train(ckpt_name=ckpt_name, prediction_method=prediction_method)

        if 'hybrid' in args.dataset or 'revised' in args.dataset:
            test_sent_result, _, test_raw_results = trainer.test(datas.test_dataloader, content_level_eval=False, prediction_method=prediction_method)
            experiment_result['test_result'] = {'line': test_sent_result, 'raw': test_raw_results}
        else:
            test_sent_result, test_content_result, test_raw_results = trainer.test(datas.test_dataloader, content_level_eval=True, prediction_method=prediction_method)
            experiment_result['test_result'] = {'line': test_sent_result, 'document': test_content_result, 'raw': test_raw_results}

    experiment_results.append(experiment_result)

    with open(f'train_cpp/result/new/experiment_results_{args.ckpt_name}.json', 'w') as file:
        json.dump(experiment_results, file, ensure_ascii=False, cls=NpEncoder)