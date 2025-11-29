import random
import httpx
import msgpack
import threading
import time
import os
import argparse
import json
import scipy
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm


def access_api(text, api_url, mask_start, mask_end, do_generate=False):
    """

    :param text: input text
    :param api_url: api
    :param do_generate: whether generate or not
    :return:
    """
    with httpx.Client(timeout=None) as client:
        post_data = {
            "text": text,
            "do_generate": do_generate,
            "mask_start":mask_start, 
            "mask_end": mask_end
        }
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)
    if prediction.status_code == 200:
        content = msgpack.unpackb(prediction.content)
    else:
        content = None
    return content


def get_features(type, input_file, output_file):
    """
    get [losses, begin_idx_list, ll_tokens_list, label_int, label] based on raw lines
    """

    incoder_api = 'http://0.0.0.0:6006/inference'
    polycoder_api = 'http://0.0.0.0:6007/inference'
    codellama_py_api = 'http://0.0.0.0:6008/inference'
    starcoder2_api = 'http://0.0.0.0:6009/inference'
    

    en_model_apis = [incoder_api, polycoder_api, codellama_py_api, starcoder2_api]
    #en_model_apis = [incoder_api]
    en_labels = {
        'human':0,
        'AI':1
    }
    
    with open(input_file, 'r') as f:
        lines = [json.loads(line) for line in f]

    print('input file:{}, length:{}'.format(input_file, len(lines)))

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in tqdm(lines):
            line = data['text']
            label = data['label']
            llm = data['LLM']
            status = data['status_in_folder']

            mask_start = data['mask_start']
            mask_end = data['mask_end']
            
            losses = []
            begin_idx_list = []
            ll_tokens_list = []
            model_apis = en_model_apis
            label_dict = en_labels

            if label == "Mixed":
                label_int = 1
                label = "AI"
            else:
                label_int = label_dict[label]

            error_flag = False
            
 
            for api in model_apis:
                print("api:", api)
                try:
                    loss, begin_word_idx, ll_tokens = access_api(line.strip(), api, mask_start, mask_end)
                except TypeError:
                    print("return NoneType, probably gpu OOM, discard this sample",api)
                    error_flag = True
                    break
                losses.append(loss)
                begin_idx_list.append(begin_word_idx)
                ll_tokens_list.append(ll_tokens)

            
            # if oom, discard this sample
            if error_flag:
                continue

            result = {
                'losses': losses,
                'begin_idx_list': begin_idx_list,
                'll_tokens_list': ll_tokens_list,
                'label_int': label_int,
                'label': label,
                # 'score': score, # python-only
                'mask_start': mask_start,
                'mask_end': mask_end,
                'eval' : data['eval'],
                'LLM': llm, # aigcodeset
                'status_in_folder': status, # aigcodeset
                'text': line,
                'problem_id': data['problem_id'],
                'user_id': "",
                'mixed': data['label'] not in en_labels
            }

            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--do_normalize", action="store_true", help="normalize the features")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    get_features(type='en', input_file=args.input_file, output_file=args.output_file)
