__author__ = "Zhuoyue WAN Russ"

import os
from random import shuffle
import pytorch_lightning as pl
import torch
import argparse
import datetime
import wandb
import math

from os.path import join
from torch.utils.data import DataLoader

from transformers import T5ForConditionalGeneration,AutoConfig,AutoTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import Dataset, load_dataset, concatenate_datasets,interleave_datasets

from model.utility import HistoryCallback, EpochProgressBar, get_lr_scheduler, get_optimizer
from model.mlm_utils import compute_input_and_target_lengths,DataCollatorForT5MLM
from model.mlm_bidir_utils import DataCollatorForT5MLM_bidir, process_dataset, process_dataset_bidir

from utilities.functions import check_create_folder,get_library, read_overpass_split, read_tags_csv,add_prefix
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, default='/root/autodl-tmp/LLM4Geo/OSMT5/pretrain')
    
    parser.add_argument('--data_dir', default='../dataset', type=str)
    parser.add_argument('--osm_dir', default='../osm', type=str)
    
    parser.add_argument('--model_type', type=str, default='codet5-small'
                        ,choices=['codet5-small','codet5-base','codet5-large',
                                    'codet5p-220m','codet5p-770m',
                                    'byt5-small','byt5-base','byt5-large'])
    parser.add_argument('--mode', type=str, default='pretrain',choices=['pretrain','debug'])
    parser.add_argument("--task", type=str, default="pretrain", help="task name",choices=['pretrain'])

    parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size') #16,8

    # exp_version
    parser.add_argument('--exp_version', type=str, default='v0', help='exp_version')

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs') # 200

    # hyper parameters
    parser.add_argument('--mlm_probability', type=float, default=0.15, help='mlm_probability')
    parser.add_argument('--mean_noise_span_length', type=int, default=3, help='mean_noise_span_length') #

    parser.add_argument('--deep_speed_type', type=str, default='deepspeed_stage_1'
                        ,choices=['deepspeed_stage_1','deepspeed_stage_2','deepspeed_stage_2_offload','deepspeed_stage_3','deepspeed_stage_3_offload'])

    parser.add_argument('--acc_grad_batches', type=int, default=1, help='accumulate_grad_batches')
    parser.add_argument('--precision', type=str, default='bf16', help='precision',choices=['bf16','16','16-mixed','32'])
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate') # 之前是5e-4 1e-4
    parser.add_argument('--warmup_rate', type=float, default=0.1, help='warmup rate')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='gradient clip value')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
    parser.add_argument('--final_cosine', type=float, default=5e-8, help='final cosine')
    
# pretrain dataset parameters
    parser.add_argument('--use_original_data', type=str2bool, default=True, help='Use original data')
    # mlm objective
    parser.add_argument('--only_MLM', type=str2bool, default=False, help='Only use MLM objective')
    parser.add_argument('--use_tags_all', type=str2bool, default=True, help='Use all tags')
    parser.add_argument('--use_descrip', type=str2bool, default=True, help='Use descriptions')
    # bidirectional objective
    parser.add_argument('--use_tag_descrip', type=str2bool, default=True, help='Use tag descriptions')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(1234)

    # precision
    if args.model_type in ['t5-small','t5-base','t5-large','t5-3b','t5-11b','byt5-small','byt5-base','byt5-large']:
        args.precision = 'bf16'
    else:
        args.precision = '16-mixed'
        
    args.adam_name = 'AdamW' if 'offload' not in args.deep_speed_type else 'DeepSpeedCPUAdam'

    print('Current working task:',args.task)
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d")
    current_time = current_time + '-' + args.exp_version
    
    MODEL_FOLDER = join(args.project_folder, 'ckpt/'+args.task +'/'+ args.model_type+'/'+current_time)
    LOGGER_FOLDER = join(args.project_folder, 'ckpt/'+args.task+'/'+ args.model_type+'/'+current_time+"/logger")
    
    # check_create_folder(CHECKPOINT_FOLDER,ask_to_rm_if_exists=False)
    check_create_folder(MODEL_FOLDER,ask_to_rm_if_exists=False)
    check_create_folder(LOGGER_FOLDER,ask_to_rm_if_exists=False)
    
    MODEL_TYPE, tokenizer_library= get_library(args.model_type)
    config = AutoConfig.from_pretrained(MODEL_TYPE)
    # model = T5ForConditionalGeneration.from_pretrained(MODEL_TYPE,config=config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE,use_fast=True)
    tokenizer.model_max_length = int(1e9)
    
    special_tokens_dict = {'additional_special_tokens': ["<NL>", "<OVQ>","<Tag>"]}

    tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)

    # We increase the input_length, because instead of masking tokens T5 replaces
    # masked spans with a single token, therefore to avoid padding we need to have
    # longer sequences at the start, before masking
    before_mask_input_length, target_length = compute_input_and_target_lengths(
        inputs_length=args.max_input_length,
        noise_density=args.mlm_probability,
        mean_noise_span_length=args.mean_noise_span_length,
    )
    # 将before_mask_input_length和target_length加入args
    args.before_mask_input_length = before_mask_input_length
    args.target_length = target_length
    args.MODEL_TYPE = MODEL_TYPE

    # Helper functions
    def create_mlm_dataset(data, prefix=None):
        if prefix:
            data = add_prefix(data, prefix)
        return {'train': Dataset.from_dict({"text": data})}

    def create_bidir_dataset(source, target, source_prefix=None, target_prefix=None):
        if source_prefix:
            source = add_prefix(source, source_prefix)
        if target_prefix:
            target = add_prefix(target, target_prefix)
        return {'train': Dataset.from_dict({"source": source, "target": target})}

    # Read data and create datasets
    syn_nl, syn_ovq = read_overpass_split(args.data_dir + "/dataset.syn")
    syn_nl_mlm = create_mlm_dataset(syn_nl)
    syn_ovq_mlm = create_mlm_dataset(syn_ovq)

    if args.use_descrip:
        _, descrip = read_tags_csv(args.osm_dir + "/tags_description.csv")
        descrip_mlm = create_mlm_dataset(descrip)

    if args.use_original_data:
        train_nl, train_ovq = read_overpass_split(args.data_dir + "/dataset.train")
        val_nl, val_ovq = read_overpass_split(args.data_dir + "/dataset.dev")  

        # 创建训练集和验证集 (MLM 模式)
        original_nl_mlm_train = create_mlm_dataset(train_nl)
        original_ovq_mlm_train = create_mlm_dataset(train_ovq)
        original_nl_mlm_val = create_mlm_dataset(val_nl)
        original_ovq_mlm_val = create_mlm_dataset(val_ovq)

    # If not only_MLM, create bidirectional datasets
    if not args.only_MLM:
        syn_data_bid = create_bidir_dataset(syn_nl, syn_ovq, source_prefix="<NL> ", target_prefix="<OVQ> ")

        if args.use_tag_descrip:
            tag_src, desc_trg = read_tags_csv(args.osm_dir + "/tags_description.csv")
            tag_desc_bid = create_bidir_dataset(tag_src, desc_trg, source_prefix="<Tag> ", target_prefix="<NL> ")

        if args.use_original_data:
            original_data_bid_train = create_bidir_dataset(train_nl, train_ovq, source_prefix="<NL> ", target_prefix="<OVQ> ")
            original_data_bid_val = create_bidir_dataset(val_nl, val_ovq, source_prefix="<NL> ", target_prefix="<OVQ> ")
    # Tokenize MLM datasets
    Syn_nl = process_dataset(syn_nl_mlm, args, tokenizer)
    Syn_ovq = process_dataset(syn_ovq_mlm, args, tokenizer)

    if args.use_descrip:
        Descrip = process_dataset(descrip_mlm, args, tokenizer)
    if args.use_original_data:
        Original_nl_train = process_dataset(original_nl_mlm_train, args, tokenizer)
        Original_ovq_train = process_dataset(original_ovq_mlm_train, args, tokenizer)

        Original_nl_val = process_dataset(original_nl_mlm_val, args, tokenizer)
        Original_ovq_val = process_dataset(original_ovq_mlm_val, args, tokenizer)

    # Tokenize bidirectional datasets (only if not only_MLM)
    if not args.only_MLM:
        Syn_data_bid = process_dataset_bidir(syn_data_bid, args, tokenizer)

        if args.use_tag_descrip:
            Tag_desc_bid = process_dataset_bidir(tag_desc_bid, args, tokenizer)

        if args.use_original_data:
            Original_data_bid_train = process_dataset_bidir(original_data_bid_train, args, tokenizer)
            Original_data_bid_val = process_dataset_bidir(original_data_bid_val, args, tokenizer)
    # Combine MLM datasets
    mlm_datasets = {
        'train': interleave_datasets([Syn_nl['train'], Syn_ovq['train']] +
                                     ([Descrip['train']] if args.use_descrip else []) +
                                     ([Original_nl_train['train'], Original_ovq_train['train']] if args.use_original_data else [])),
        'val': interleave_datasets([Syn_nl['train'], Syn_ovq['train']] +
                                      ([Descrip['train']] if args.use_descrip else []) +
                                      ([Original_nl_val['train'], Original_ovq_val['train']] if args.use_original_data else [])
                                      )
    }

    # Combine bidirectional datasets if not only_MLM
    if not args.only_MLM:
        bidir_datasets = {
            'train': interleave_datasets([Syn_data_bid['train']] +
                                         ([Tag_desc_bid['train']] if args.use_tag_descrip else []) +
                                         ([Original_data_bid_train['train']] if args.use_original_data else [])),
            'val': interleave_datasets([Syn_data_bid['train']] +
                                        ([Tag_desc_bid['train']] if args.use_tag_descrip else []) +
                                        ([Original_data_bid_val['train']] if args.use_original_data else [])
                                        )
        }

        # Combine MLM and bidirectional datasets
        mixed_split = {
            'train': interleave_datasets([mlm_datasets['train'], bidir_datasets['train']]),
            'val': interleave_datasets([mlm_datasets['val'], bidir_datasets['val']])
        }
    else:
        # If only MLM, use only the mlm_datasets
        mixed_split = mlm_datasets

    args.training_set_len = len(mixed_split['train'])
    args.batches_per_epoch = math.ceil(args.training_set_len / args.batch_size)

    # Dynamically choose the data collator based on only_MLM
    if args.only_MLM:
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=args.mlm_probability,
            mean_noise_span_length=args.mean_noise_span_length,
            input_length=args.max_input_length,
            target_length=args.target_length,
            pad_token_id=config.pad_token_id,
        )
    else:
        data_collator = DataCollatorForT5MLM_bidir(
            tokenizer=tokenizer,
            noise_density=args.mlm_probability,
            mean_noise_span_length=args.mean_noise_span_length,
            input_length=args.max_input_length,
            target_length=args.target_length,
            pad_token_id=config.pad_token_id,
        )

    # DataLoader
    train_dataloader = DataLoader(
        mixed_split['train'],
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=False,
    )
    
    # DataLoader for Validation
    val_dataloader = DataLoader(
        mixed_split['val'],
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=False,
    )

    # Define the model
    class T5PreTrainer(pl.LightningModule):
        def __init__(self, args):
            super(T5PreTrainer, self).__init__()
            self.args = args
            self.save_hyperparameters(args)
            self.config = AutoConfig.from_pretrained(args.MODEL_TYPE)
            self.model = T5ForConditionalGeneration.from_pretrained(args.MODEL_TYPE, config=self.config,
                                                                    force_download=False)
            self.model.resize_token_embeddings(len(tokenizer))

        def forward(self, input_ids, attention_mask, labels):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return outputs

        def training_step(self, batch, batch_idx):
            batch['attention_mask'] = batch['input_ids'] != self.config.pad_token_id
            outputs = self(**batch)
            loss = outputs.loss
            self.log("training_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            last_lr = self.lr_schedulers().get_last_lr()[0]
            self.log("learning_rate", last_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            batch['attention_mask'] = batch['input_ids'] != self.config.pad_token_id
            outputs = self(**batch)
            loss = outputs.loss
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return loss

        def compute_warmup_steps(self):
            total_training_steps = self.args.num_epochs * self.args.batches_per_epoch
            warmup_steps = int(total_training_steps * self.args.warmup_rate)
            return warmup_steps, total_training_steps

        def configure_optimizers(self):
            warmup_steps, total_training_steps = self.compute_warmup_steps()
            optimizer = get_optimizer(self.args, self.parameters())
            lr_scheduler = get_lr_scheduler('linear', optimizer, warmup_steps, total_training_steps, final_cosine=self.args.final_cosine)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        def train_dataloader(self):
            return train_dataloader
        
        def val_dataloader(self):
            return val_dataloader

    # Initialize model and logger
    pt_model = T5PreTrainer(args)
    wandb_logger = WandbLogger(
        dir=LOGGER_FOLDER,
        save_dir=LOGGER_FOLDER,
        name=f"{args.task}_{args.model_type}_{now}",
        project="OSMT5",
        group=args.task,
        config=args,
    )

    # Callbacks
    progress_bar = EpochProgressBar()
    history = HistoryCallback()
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='training_loss',
    #     dirpath=MODEL_FOLDER,
    #     filename='model_best',
    #     save_top_k=1,
    #     mode='min'
    # )
    # # Early Stopping
    # early_stopping_callback = EarlyStopping(
    #     monitor="val_loss",  # 监控验证损失
    #     patience=3,  # 如果验证损失连续 3 个 epoch 无改进，则停止训练
    #     mode="min",  # 希望验证损失越小越好
    #     verbose=True,
    # )
    
    # Trainer
    trainer = Trainer(
        default_root_dir=MODEL_FOLDER,
        callbacks=[progress_bar, history],
        logger=[wandb_logger],
        max_epochs=args.num_epochs,
        limit_train_batches=args.batches_per_epoch,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.acc_grad_batches,
        strategy=args.deep_speed_type,
    )

    trainer.fit(pt_model)

    # Save model and tokenizer
    pt_model.model.save_pretrained(MODEL_FOLDER)
    tokenizer.save_pretrained(join(MODEL_FOLDER, "tokenizer"))
    wandb_logger.experiment.finish()