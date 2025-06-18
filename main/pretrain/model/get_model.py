import torch
from os.path import join

from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    AutoConfig,
)
from .t5_model_support_functions import expand_tokenizer

# Load Model Functions

def rename_model_type(args):
    '''
    add 'Salesforce/' to model_type
    example: codet5p-2b -> Salesforce/codet5p-2b
    '''
    prefix, _, size = args.model_type.partition('-')
    if prefix in ['codet5', 'codet5p']:
        MODEL_TYPE = 'Salesforce/' + args.model_type
    else:
        MODEL_TYPE = args.model_type
    return MODEL_TYPE

def get_config(args):
    MODEL_TYPE = rename_model_type(args)
    config = AutoConfig.from_pretrained(MODEL_TYPE)
    return config

def get_tokenizer(args):
    MODEL_TYPE = rename_model_type(args)
    if args.use_original_ckpt:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_TYPE,
            use_fast=True
        )
        tokenizer.model_max_length = int(1e9)
        
        special_tokens_dict = {'additional_special_tokens': ["<NL>", "<VQL>","<schema>","<Question>","<Answer>","<Table>"]}
        tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)
        # new_tokens = ["<NL>", "<VQL>","<schema>","<Question>","<Answer>","<MLM>","<Table>"]
        # expand_tokenizer(new_tokens, tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(join(args.ckpt_path, "tokenizer"))
    return tokenizer

def get_model(args, config, tokenizer):
    MODEL_TYPE = rename_model_type(args)
    if args.use_original_ckpt:
        model = T5ForConditionalGeneration.from_pretrained(
            MODEL_TYPE,
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.ckpt_path)

        model.resize_token_embeddings(len(tokenizer))
        # model.load_state_dict(torch.load(args.ckpt_path), strict=True)
        
        torch.cuda.empty_cache()

    return model


