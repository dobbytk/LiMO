import numpy as np
import os
import random
import torch
import json
import pandas as pd
import pickle

from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import normalize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DataManager:

    def __init__(self, datas, batch_size, max_len, human_label, id2label, word_pad_idx=0, label_pad_idx=-1):
        set_seed(0)
        self.batch_size = batch_size
        self.max_len = max_len
        self.human_label = human_label
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        data = dict()
        train_dict = self.initialize_dataset(datas[0])
        data["train"] = Dataset.from_dict(train_dict)
        valid_dict = self.initialize_dataset(datas[1])
        data["valid"] = Dataset.from_dict(valid_dict)
        test_dict = self.initialize_dataset(datas[2])
        data["test"] = Dataset.from_dict(test_dict)
        datasets = DatasetDict(data)
        self.data = datasets
        self.train_dataloader = self.get_train_dataloader(datasets["train"])
        self.val_dataloader = self.get_eval_dataloader(datasets["valid"])
        self.test_dataloader = self.get_eval_dataloader(datasets["test"])
        # from IPython import embed
        # embed()

    def initialize_dataset(self, total_samples):
        samples_dict = {'features': [],'ccfeatures': [], 'mask_start': [], 'mask_end':[], 'label_int':[], 'label': [], 'text': [], 'user_id': [], 'problem_id':[], 'line_count':[]}

        for item in tqdm(total_samples):
            text = item['text'].splitlines(True)
            label = item['label']
            
            mask_start = item['mask_start']
            mask_end = item['mask_end']
            user_id = item['user_id']
            problem_id = item['problem_id']
            
            label_int = item['label_int']
            
            ccfeature_list = item['ccfeature']
            begin_idx_list = item['begin_idx_list']
            ll_tokens_list = item['ll_tokens_list']

            begin_idx_list = np.array(begin_idx_list)
            # Get the maximum value in begin_idx_list, which indicates where we need to truncate.
            max_begin_idx = np.max(begin_idx_list)
            # Truncate all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[max_begin_idx:]
                
            # Get the length of all vectors and take the minimum
            min_len = np.min([len(ll_tokens) for ll_tokens in ll_tokens_list])
            # Align the lengths of all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[:min_len]
            if len(ll_tokens_list) == 0 or len(ll_tokens_list[0]) == 0:
                continue
            ll_tokens_list = np.array(ll_tokens_list)
            # ll_tokens_list = normalize(ll_tokens_list, norm='l1')
            ll_tokens_list = ll_tokens_list.transpose()
            ll_tokens_list = ll_tokens_list.tolist()
            
            mask_start = max(mask_start - max_begin_idx, 0)
            mask_end = min(max(mask_end - max_begin_idx, 0), min_len)
            ccfeature = ccfeature_list[max_begin_idx:min_len]
            # import IPython; IPython.embed()
            if len(ccfeature) == 0 or len(ccfeature[0]) == 0:
                continue
            
            samples_dict['features'].append(ll_tokens_list)
            samples_dict['ccfeatures'].append(ccfeature)
            samples_dict['mask_start'].append(mask_start)
            samples_dict['mask_end'].append(mask_end)
            samples_dict['label_int'].append(label_int)
            samples_dict['label'].append(label)
            samples_dict['text'].append(''.join(text[max_begin_idx:min_len]))
            samples_dict['user_id'].append(user_id)
            samples_dict['problem_id'].append(problem_id)
            samples_dict['line_count'].append(len(text))
        return samples_dict


    def get_train_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=RandomSampler(dataset),
                          collate_fn=self.data_collator)
    
    def get_eval_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=SequentialSampler(dataset),
                          collate_fn=self.data_collator)
    
    def pad_ccfeatures_numpy(self, ccfeatures, max_len=1024, pad_value=-1):
        pad_value=self.label_pad_idx
        ccfeatures = [np.array(f, dtype=np.float32) for f in ccfeatures]
    
        lengths = np.array([len(f) for f in ccfeatures])
    
        padded_ccfeatures = np.array([
            np.pad(f, ((0, max_len - len(f)), (0, 0)), mode='constant', constant_values=pad_value) if len(f) < max_len else f[:max_len] 
            for f in ccfeatures
        ])
        
        return padded_ccfeatures
    
    def data_collator(self, samples):
        batch = {}

        problem_id = [sample['problem_id'] for sample in samples]
        user_id = [sample['user_id'] for sample in samples]
        ccfeatures = [sample['ccfeatures'] for sample in samples]
        features = [sample['features'] for sample in samples]
        mask_start = [sample['mask_start'] for sample in samples]
        mask_end = [sample['mask_end'] for sample in samples]
        text = [sample['text'] for sample in samples]
        label = [sample['label'] for sample in samples]
        features, masks = self.process_and_convert_to_tensor(features)
        # pad_masks = ~masks * -1
        pad_masks = (1 - masks) * self.label_pad_idx
        
        for idx, (m_start, m_end) in enumerate(zip(mask_start, mask_end)):
            lines = text[idx].split('\n')
            prefix_len = m_start
            if prefix_len > self.max_len:
                prefix_ids = self.sequence_labels_to_ids(self.max_len, self.human_label)
                if masks[idx] != None and prefix_ids != None:
                    masks[idx][:] = prefix_ids[:]
                continue
            
            total_len = len(lines)
            response_len = m_end + 1 - m_start
            
            if prefix_len > 0:
                prefix_ids = self.sequence_labels_to_ids(prefix_len, self.human_label)
                
                if masks[idx] != None and prefix_ids != None:
                    masks[idx][:prefix_len] = prefix_ids[:]
            if total_len - prefix_len > 0:
                if prefix_len + response_len >= self.max_len:
                    machine_ids = self.sequence_labels_to_ids(self.max_len - prefix_len, label[idx])
                    
                    if masks[idx] != None and machine_ids != None:
                        masks[idx][prefix_len:] = machine_ids[:]
                    continue                    
                else:
                    pre_res_len = response_len + prefix_len
                    if pre_res_len >= total_len:
                        machine_ids = self.sequence_labels_to_ids(total_len - prefix_len, label[idx])
                        if masks[idx] != None and machine_ids != None:
                            masks[idx][prefix_len:total_len] = machine_ids[:]
                    else:
                        machine_ids = self.sequence_labels_to_ids(response_len, label[idx])
                        if masks[idx] != None and machine_ids != None:
                            masks[idx][prefix_len:pre_res_len] = machine_ids[:]
                
                        if total_len >= self.max_len:
                            human_ids = self.sequence_labels_to_ids(self.max_len - pre_res_len, self.human_label)
                            if masks[idx] != None and human_ids != None:
                                masks[idx][pre_res_len:self.max_len] = human_ids[:]
                        else:
                            human_ids = self.sequence_labels_to_ids(total_len - pre_res_len, self.human_label)
                            
                            if masks[idx] != None and human_ids != None:
                                masks[idx][pre_res_len:total_len] = human_ids[:]
            if total_len < len(masks[idx]):
                masks[idx][total_len:] = -1
            else:
                masks[idx] += pad_masks[idx]
        padded_ccfeatures = self.pad_ccfeatures_numpy(ccfeatures)
            
        batch['features'] = features
        batch['ccfeatures'] = torch.tensor(padded_ccfeatures, dtype=torch.float32)
        batch['labels'] = masks
        batch['text'] = text
        batch['problem_id'] = problem_id
        batch['user_id'] = user_id

        return batch

    
    def sequence_labels_to_ids(self, seq_len, label):
        prefix = ['B-', 'M-', 'E-', 'S-']
        if seq_len <= 0:
            return None
        elif seq_len == 1:
            label = 'S-' + label
            return torch.tensor([self.label2id[label]], dtype=torch.long)
        else:
            ids = []
            ids.append(self.label2id['B-'+label])
            ids.extend([self.label2id['M-'+label]] * (seq_len - 2))
            ids.append(self.label2id['E-'+label])
            return torch.tensor(ids, dtype=torch.long)

    def process_and_convert_to_tensor(self, data):
        """ here, data is features. """
        max_len = self.max_len
        # data shape: [B, S, E]
        feat_dim = len(data[0][0])
        padded_data = [  # [[0] * feat_dim] + 
            seq + [[0] * feat_dim] * (max_len - len(seq)) for seq in data
        ]
        padded_data = [seq[:max_len] for seq in padded_data]
        masks = [[1] * min(len(seq), max_len) + [0] *
                (max_len - min(len(seq), max_len)) for seq in data]
        
        tensor_data = torch.tensor(padded_data, dtype=torch.float)
        tensor_mask = torch.tensor(masks, dtype=torch.long)
        return tensor_data, tensor_mask


    def _split_en_sentence(self, sentence, use_sp=False):
        import re
        pattern = re.compile(r'\S+|\s')
        words = pattern.findall(sentence)
        if use_sp:
            words = ["▁" if item == " " else item for item in words]
        return words

    
    def _split_code_sentence(self, code, use_sp=False):
        import re
        pattern = re.compile(
        r'"""|\'\'\'|"|\'|#|==|'
        r'\n|'
        r'[^\S\n]+|'
        r'\w+|[.,()\[\]{};:\=\_\+\-\*\/\~\!\%\^\&\<\>\?]')
        tokens = pattern.findall(code)
        return tokens
        

    
    def split_sentence(self, sentence, use_sp=False, cn_percent=0):
        return self._split_code_sentence(sentence, use_sp)



class DataManagerTest:

    def __init__(self, datas, batch_size, max_len, human_label, id2label, word_pad_idx=0, label_pad_idx=-1):
        set_seed(0)
        self.batch_size = batch_size
        self.max_len = max_len
        self.human_label = human_label
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        data = dict()
        train_dict = self.initialize_dataset(datas[0])
        data["train"] = Dataset.from_dict(train_dict)
        valid_dict = self.initialize_dataset(datas[1])
        data["valid"] = Dataset.from_dict(valid_dict)
        test_dict = self.initialize_dataset(datas[2])
        data["test"] = Dataset.from_dict(test_dict)
        
        datasets = DatasetDict(data)
        self.data = datasets
        self.train_dataloader = self.get_train_dataloader(datasets["train"])
        self.val_dataloader = self.get_eval_dataloader(datasets["valid"])
        self.test_dataloader = self.get_eval_dataloader(datasets["test"])
        

    def initialize_dataset(self, total_samples):
        samples_dict = {'features': [],'ccfeatures': [], 'mask': [], 'label_int':[], 'label': [], 'text': [], 'user_id': [], 'problem_id':[]}

        for item in tqdm(total_samples):
            text = item['text']
            label = item['label']
            mask = item['mask']
            #mask_start = item['mask_start']
            #mask_end = item['mask_end']
            user_id = 0 #item['annotator_id']#item['user_id']
            problem_id = item['problem_id']
            label_int = item['label_int']
            
            ccfeature_list = item['ccfeature']
            begin_idx_list = item['begin_idx_list']
            ll_tokens_list = item['ll_tokens_list']

            begin_idx_list = np.array(begin_idx_list)
            # Get the maximum value in begin_idx_list, which indicates where we need to truncate.
            max_begin_idx = np.max(begin_idx_list)
            # Truncate all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[max_begin_idx:]
            # Get the length of all vectors and take the minimum
            min_len = np.min([len(ll_tokens) for ll_tokens in ll_tokens_list])
            # Align the lengths of all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[:min_len]
            if len(ll_tokens_list) == 0 or len(ll_tokens_list[0]) == 0:
                continue
            ll_tokens_list = np.array(ll_tokens_list)
            # ll_tokens_list = normalize(ll_tokens_list, norm='l1')
            ll_tokens_list = ll_tokens_list.transpose()
            ll_tokens_list = ll_tokens_list.tolist()
            
            samples_dict['features'].append(ll_tokens_list)
            samples_dict['ccfeatures'].append(ccfeature_list)
            samples_dict['mask'].append(mask)
            samples_dict['label_int'].append(label_int)
            samples_dict['label'].append(label)
            samples_dict['text'].append(text)
            samples_dict['user_id'].append(user_id)
            samples_dict['problem_id'].append(problem_id)
        return samples_dict


    def get_train_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=RandomSampler(dataset),
                          collate_fn=self.data_collator)
    
    def get_eval_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          sampler=SequentialSampler(dataset),
                          collate_fn=self.data_collator)
    
    def pad_ccfeatures_numpy(self, ccfeatures, max_len=1024, pad_value=-1):
        pad_value=self.label_pad_idx
        ccfeatures = [np.array(f, dtype=np.float32) for f in ccfeatures]
        lengths = np.array([len(f) for f in ccfeatures])
        padded_ccfeatures = np.array([
            np.pad(f, ((0, max_len - len(f)), (0, 0)), mode='constant', constant_values=pad_value) if len(f) < max_len else f[:max_len] 
            for f in ccfeatures
        ])
        
        return padded_ccfeatures
    
    def data_collator(self, samples):
        # samples: {'features': [], 'prompt_len': [], 'label': [], 'text': []}
        # batch: {'features': [], 'labels': [], 'text': []}
        batch = {}

        ccfeatures = [sample['ccfeatures'] for sample in samples]
        features = [sample['features'] for sample in samples]
        mask_label = [sample['mask'] for sample in samples]
        text = [sample['text'] for sample in samples]
        label = [sample['label'] for sample in samples]
        features, masks = self.process_and_convert_to_tensor(features)
        # pad_masks = ~masks * -1
        pad_masks = (1 - masks) * self.label_pad_idx
        
        for idx, mask_labels in enumerate(mask_label):
            lines = text[idx].split('\n')
            total_len = len(lines)

            current_segment = []
            current_label = None

            if total_len > self.max_len:
                mask_labels = mask_labels[:self.max_len]

            if masks[idx] != None:
                # mask_labels 리스트를 순차적으로 처리하며 연속된 세그먼트를 처리
                for i, l in enumerate(mask_labels):
                    if l != current_label:  # 새로운 세그먼트가 시작되는 경우
                        if current_segment:  # 이전 세그먼트가 있으면 처리
                            segment_len = len(current_segment)
                            if segment_len > 0:
                                # 현재 라벨에 맞는 시퀀스 라벨을 생성
                                if current_label == 1:  # machine label
                                    machine_ids = self.sequence_labels_to_ids(segment_len, label[idx])
                                    masks[idx][current_segment[0]:current_segment[-1] + 1] = machine_ids[:]
                                else:  # human label
                                    human_ids = self.sequence_labels_to_ids(segment_len, self.human_label)
                                    masks[idx][current_segment[0]:current_segment[-1] + 1] = human_ids[:]

                        current_segment = [i]
                        current_label = l
                    else:
                        current_segment.append(i)

                # 마지막 세그먼트 처리
                if current_segment:
                    segment_len = len(current_segment)
                    if segment_len > 0:
                        if current_label == 1:  # machine label
                            machine_ids = self.sequence_labels_to_ids(segment_len, label[idx])
                            masks[idx][current_segment[0]:current_segment[-1] + 1] = machine_ids[:]
                        else:  # human label
                            human_ids = self.sequence_labels_to_ids(segment_len, self.human_label)
                            masks[idx][current_segment[0]:current_segment[-1] + 1] = human_ids[:]

            # padding 처리
            if total_len < self.max_len:
                masks[idx][total_len:self.max_len] = -1 #pad_masks[idx][total_len:self.max_len]
        padded_ccfeatures = self.pad_ccfeatures_numpy(ccfeatures)

        batch['features'] = features
        batch['ccfeatures'] = torch.tensor(padded_ccfeatures, dtype=torch.float32)
        batch['labels'] = masks
        batch['text'] = text

        return batch

    
    def sequence_labels_to_ids(self, seq_len, label):
        prefix = ['B-', 'M-', 'E-', 'S-']
        if seq_len <= 0:
            return None
        elif seq_len == 1:
            label = 'S-' + label
            return torch.tensor([self.label2id[label]], dtype=torch.long)
        else:
            ids = []
            ids.append(self.label2id['B-'+label])
            ids.extend([self.label2id['M-'+label]] * (seq_len - 2))
            ids.append(self.label2id['E-'+label])
            return torch.tensor(ids, dtype=torch.long)

    def process_and_convert_to_tensor(self, data):
        """ here, data is features. """
        max_len = self.max_len
        # data shape: [B, S, E]
        feat_dim = len(data[0][0])
        padded_data = [  # [[0] * feat_dim] + 
            seq + [[0] * feat_dim] * (max_len - len(seq)) for seq in data
        ]
        padded_data = [seq[:max_len] for seq in padded_data]
        masks = [[1] * min(len(seq), max_len) + [0] *
                (max_len - min(len(seq), max_len)) for seq in data]

        tensor_data = torch.tensor(padded_data, dtype=torch.float)
        tensor_mask = torch.tensor(masks, dtype=torch.long)
        return tensor_data, tensor_mask


    def _split_en_sentence(self, sentence, use_sp=False):
        import re
        pattern = re.compile(r'\S+|\s')
        words = pattern.findall(sentence)
        if use_sp:
            words = ["▁" if item == " " else item for item in words]
        return words

    
    def _split_code_sentence(self, code, use_sp=False):
        import re
        pattern = re.compile(
        r'"""|\'\'\'|"|\'|#|==|'
        r'\n|'
        r'[^\S\n]+|'
        r'\w+|[.,()\[\]{};:\=\_\+\-\*\/\~\!\%\^\&\<\>\?]')
        
        tokens = pattern.findall(code)
        return tokens
    
    def split_sentence(self, sentence, use_sp=False, cn_percent=0.2):
        return self._split_code_sentence(sentence, use_sp)
