import torch
import transformers
import numpy as np

from backend_utils import TokenizerPPLCalc

# mosec
from mosec import Worker
from mosec.mixin import MsgpackMixin
# llama
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


class BaseModel(MsgpackMixin, Worker):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = None
        self.base_model = None
        self.generate_len = 1024
        self.mask_start = None
        self.mask_end = None

    def forward_calc_ppl(self):
        pass

    def forward_gen(self):
        #torch.cuda.empty_cache()
        self.base_tokenizer.padding_side = 'left'
        # 1. single generate
        if isinstance(self.text, str):
            tokenized = self.base_tokenizer(self.text, return_tensors="pt").to(
                self.device)
            tokenized = tokenized.input_ids
            gen_tokens = self.base_model.generate(tokenized,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_tokens = gen_tokens.squeeze()
            result = self.base_tokenizer.decode(gen_tokens.tolist())
            return result
        # 2. batch generate
        # msgpack.unpackb(self.text, use_list=False) == tuple
        elif isinstance(self.text, tuple):
            inputs = self.base_tokenizer(self.text,
                                         padding=True,
                                         return_tensors="pt").to(self.device)
            gen_tokens = self.base_model.generate(**inputs,
                                                  do_sample=True,
                                                  max_length=self.generate_len)
            gen_texts = self.base_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            return gen_texts

    def forward(self, data):
        """
        :param data: ['text': str, "do_generate": bool]
        :return:
        """
        self.text = data["text"]
        self.mask_start = data["mask_start"]
        self.mask_end = data["mask_end"]
        
        self.do_generate = data["do_generate"]
        if self.do_generate:
            return self.forward_gen()
        else:
            return self.forward_calc_ppl()

class PolyCoderModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'NinedayWang/PolyCoder-160M')
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'NinedayWang/PolyCoder-160M')
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model.to(self.device)
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = TokenizerPPLCalc(
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device, 
                                                   'polycoder')

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text, self.mask_start, self.mask_end)

class CodeLlamaModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'codellama/CodeLlama-7b-hf')
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'codellama/CodeLlama-7b-hf')
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        
        self.base_model.to(self.device)
        self.ppl_calculator = TokenizerPPLCalc(
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device, 
                                                   'codellama'
                                                   )

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text, self.mask_start, self.mask_end)


class InCoderModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'facebook/incoder-1B')
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'facebook/incoder-1B')
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model.to(self.device)
        
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = TokenizerPPLCalc(
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device,
                                                   'incoder')

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text, self.mask_start, self.mask_end)


class StarCoder2Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'bigcode/starcoder2-3b')
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            'bigcode/starcoder2-3b')
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        self.base_model.to(self.device)
        
        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = TokenizerPPLCalc(
                                                   self.base_model,
                                                   self.base_tokenizer,
                                                   self.device, 
                                                   'starcoder2')

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = 'right'
        return self.ppl_calculator.forward_calc_ppl(self.text, self.mask_start, self.mask_end)