import re
import torch
import numpy as np
import unicodedata



class TokenizerPPLCalc(object):
    def __init__(self, base_model, base_tokenizer, device, model_name):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device
        self.model_name = model_name
        self.max_token = 1024
        
        
    def calc_sent_ppl(self, lm_logits, labels):
        """
        :param outputs: language model's output.
        :param labels: token ids.
        :return: sentence ppl, list of subtoken ppl.        
        """
        labels = labels.to(self.device)
        
        if lm_logits.size(0) > 1:
            lm_logits = lm_logits.view(1, -1, lm_logits.size(-1))
        lm_logits = lm_logits.squeeze()
        
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        ll = loss_func(shift_logits, shift_labels.view(-1))
        loss = ll.mean().item()
        ll = ll.tolist()
        return loss, ll
    
    
    
    def calc_token_ppl(self, bbs_to_words, bbs_ll):
        """
        :param bbs_to_words: list of bytes_to_words index.
        :param bbs_ll: list of bytes ppl.
        :return: list of token ppl.
        """
        start = 0
        ll_tokens = []
        while start < len(bbs_to_words) and start < len(bbs_ll):
            end = start + 1
            while end < len(
                    bbs_to_words) and bbs_to_words[end] == bbs_to_words[start]:
                end += 1
            if end > len(bbs_ll):
                break
            ll_token = bbs_ll[start:end]
            ll_tokens.append(np.mean(ll_token))
            start = end
        
        return ll_tokens
 
    
    def line_level_indexs(self, lines):
        
        line_level_tokens = self.base_tokenizer(lines)["input_ids"]
        tot_input_ids = []
        line_token_lengths = []
            
        for idx, input_ids in enumerate(line_level_tokens):    
            if idx != 0 and self.model_name in ["incoder", "codellama"]:
                input_ids = input_ids[1:]
            
            tot_input_ids += input_ids
            line_token_lengths += [len(input_ids) if len(input_ids) > 0 else 1]
       
        line_token_indexs = []
        for i in range(len(line_token_lengths)):
            line_token_indexs += [i]*line_token_lengths[i]
            
        

        return tot_input_ids, line_token_indexs 
    
    
    def forward_calc_ppl(self, text, mask_start, mask_end):
        
        torch.cuda.empty_cache()
        
        # Here we generate token indices at the line level
        lines = text.strip().splitlines(True)#text.split('\n') #split_sentence(text, use_sp=True)
        # for _ in range(len(lines)):
        #     if lines[0] == "\n":
        #         lines = lines[1:]
                
        #     else:
        #         break

        input_ids, line_token_indexs = self.line_level_indexs(lines)
        

        line_token_indexs = line_token_indexs[1:]
        print(f"인덱싱 후: {len(line_token_indexs)}")

        input_ids = torch.tensor(input_ids).unsqueeze(0)

        labels = input_ids
        
        if input_ids.size(-1) > 1024:
            print("========================")
            print(input_ids.size())
            total_token = input_ids.size(-1)
            
            slide_num = total_token//self.max_token
            
            outputs = []

            for st in range(0, total_token, self.max_token):
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    ed = st + self.max_token
                    output = self.base_model(input_ids=input_ids[:, st:ed].to(self.device)).logits.squeeze(0).detach().cpu()
                    outputs.append(output)
                    torch.cuda.empty_cache()
            lm_logits = torch.cat(outputs, dim=0).to(self.device)
            del outputs
        else:
            with torch.no_grad():
                lm_logits = self.base_model(input_ids=input_ids.to(self.device)).logits

        
        loss, ll = self.calc_sent_ppl(lm_logits, labels.to(self.device))        
        print("[ll]")
        print(len(ll))
        ll_tokens = self.calc_token_ppl(line_token_indexs, ll)
        print(f"#lines in input text: {len(lines)}")
        print("[ll_tokens]")
        print(f"#ll_tokens: {len(ll_tokens)}")
        print(ll_tokens)
        begin_word_idx = 0
        return [loss, begin_word_idx, ll_tokens]