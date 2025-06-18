from cgi import test
import os
import argparse
import shutil
import random
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

from utils import OverpassDataset
from utils import read_overpass_split, setup_seed, get_comment_queries, read_kv_split

from evaluation import get_exact_match, get_oqo_score, OverpassXMLConverter

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--data_dir', default='../dataset', type=str)
parser.add_argument('--exp_name', default='debug', type=str) # e30_lr0004
parser.add_argument('--output_dir', default='./outputs', type=str)
parser.add_argument('--model_name', default='Salesforce/codet5-base', type=str,choices=['Salesforce/codet5-small', 'Salesforce/codet5-base', 'Salesforce/codet5-large',
                                                                                        'Salesforce/codet5p-220m', 'Salesforce/codet5p-770m',
                                                                                        'google/byt5-small','google/byt5-base', 'google/byt5-large',])

parser.add_argument('--ckpt_path', default=None, type=str)

parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--batch_size', default=16, type=int) # 32 Salesforce/codet5p-220m 
parser.add_argument('--eval_batch_size', default=64, type=int)
parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
parser.add_argument('--max_length', default=600, type=int)  # max_length=600 for byt5?
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.1, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--learning_rate', default=4e-4, type=float)
parser.add_argument('--lr_scheduler_type', default='linear', type=str)
parser.add_argument('--converter_url', default="http://localhost:12346/api/convert", type=str)

parser.add_argument('--use_prefix', type=str2bool, default=False, help='Use prefix')
parser.add_argument('--use_comments_task', type=str2bool, default=False, help='Use comments task (if true, bs/2)')

parser.add_argument('--use_syn_data', type=str2bool, default=False, help='Use synthetic data')
parser.add_argument('--only_syn_data', type=str2bool, default=False, help='Only use synthetic data')

parser.add_argument('--use_key_value', type=str2bool, default=False, help='Use key-value pairs')
parser.add_argument('--use_primary_kv', type=str2bool, default=True, help='Use primary key-value pairs')

parser.add_argument('--comment_max_count', type=int, default=10, help='Maximum number of comments')
parser.add_argument('--comment_min_content_ratio', type=float, default=0.80, help='Minimum content ratio for comments')

parser.add_argument('--use_fp16', type=str2bool, default=False, help='Use FP16 precision')
parser.add_argument('--use_bf16', type=str2bool, default=True, help='Use BF16 precision')
# parser.add_argument("--deepspeed", type=str, default='configs/ds_config_zero2.json', help="Path to deepspeed config file.")

opts = parser.parse_args()
setup_seed(opts.seed)

group = 'train'
print('config', opts)

# converter = OverpassXMLConverter(url=opts.converter_url, save_frequency=-1)


def main():
    if opts.model_name == 'google/byt5-small' or opts.model_name == 'google/byt5-base' or opts.model_name == 'google/byt5-large':
        # opts.max_length = 600
        opts.use_fp16 = False
        opts.use_bf16 = True
        if opts.model_name == 'google/byt5-small':
            eval_steps = 400
        else:
            eval_steps = 800
    else:
        opts.use_fp16 = True
        opts.use_bf16 = False
        eval_steps = 800
        
    if opts.ckpt_path:
        model = T5ForConditionalGeneration.from_pretrained(opts.ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(opts.ckpt_path+"/tokenizer")
        model.config.max_length = opts.max_length
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(opts.model_name)
        model = T5ForConditionalGeneration.from_pretrained(opts.model_name)
        model.config.max_length = opts.max_length
        
    if opts.use_prefix:
        prefix_nl = '<NL> '
        # prefix_query = '<OVQ> '
        prefix_query = ''
    else:
        prefix_nl = ''
        prefix_query = ''
        
    train_texts, train_labels = read_overpass_split(opts.data_dir + "/dataset.train", prefix_nl, prefix_query)
    val_texts, val_labels = read_overpass_split(opts.data_dir + "/dataset.dev", prefix_nl, prefix_query)
    test_texts, test_labels = read_overpass_split(opts.data_dir + "/dataset.test", prefix_nl, prefix_query)

    if opts.use_key_value:  
        if opts.use_prefix:
            prefix_tag = '<Tag> '
        else:
            prefix_tag = ''
        train_kv_texts = read_kv_split(opts.data_dir + "/dataset.train", primary_kv=opts.use_primary_kv, prefix=prefix_tag)
        val_kv_texts = read_kv_split(opts.data_dir + "/dataset.dev", primary_kv=opts.use_primary_kv, prefix=prefix_tag)
        test_kv_texts = read_kv_split(opts.data_dir + "/dataset.test", primary_kv=opts.use_primary_kv, prefix=prefix_tag)
        # add key value to the last token of the input
        train_texts = [f"{t} {kv}" for t, kv in zip(train_texts, train_kv_texts)]
        val_texts = [f"{t} {kv}" for t, kv in zip(val_texts, val_kv_texts)]
        test_texts = [f"{t} {kv}" for t, kv in zip(test_texts, test_kv_texts)]
    
    comments_dataset = None
    if opts.use_comments_task:
        print('use comments task')
        comments_texts, comments_labels = get_comment_queries(opts.data_dir + '/comments.jsonl',
                                                              comment_max_count=opts.comment_max_count,
                                                              comment_min_content_ratio=opts.comment_min_content_ratio)
        comments_dataset = OverpassDataset(comments_texts, comments_labels, tokenizer, 'train',model_type=opts.model_name)
    
    if opts.use_syn_data:
        print('Using trianing data with syn data')
        syn_texts, syn_labels = read_overpass_split(opts.data_dir + "/dataset.syn", prefix_nl, prefix_query)
        if opts.only_syn_data:
            print('only finetuning by syn data')
            train_texts = syn_texts
            train_labels = syn_labels
        else:
            train_texts += syn_texts
            train_labels += syn_labels

    train_dataset = OverpassDataset(train_texts, train_labels, tokenizer, 'train', comments_dataset=comments_dataset, model_type=opts.model_name)
    val_dataset = OverpassDataset(val_texts, val_labels, tokenizer, 'val', model_type=opts.model_name)
    test_dataset = OverpassDataset(test_texts, test_labels, tokenizer, 'test', model_type=opts.model_name) 
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,padding='longest')

    output_name = f"{opts.model_name.split('/')[1]}_{opts.exp_name}"
    output_path = os.path.join(opts.output_dir, group, output_name)
    args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        evaluation_strategy="epoch",
        eval_steps=eval_steps,
        learning_rate=opts.learning_rate,  # maybe 0.0003
        lr_scheduler_type=opts.lr_scheduler_type,
        warmup_ratio=opts.warmup,  # maybe 0.1
        # per_device_train_batch_size=opts.batch_size,
        # per_device_eval_batch_size=opts.eval_batch_size,
        per_device_train_batch_size=int(opts.batch_size / opts.gradient_accumulation_steps),
        gradient_accumulation_steps=opts.gradient_accumulation_steps,
        per_device_eval_batch_size=int(opts.eval_batch_size / opts.gradient_accumulation_steps),
        weight_decay=opts.weight_decay,
        save_total_limit=2,
        load_best_model_at_end=True,
        num_train_epochs=opts.epochs,
        save_strategy="epoch",
        max_grad_norm=opts.max_grad_norm,  # maybe 0.5
        predict_with_generate=True,
        bf16=opts.use_bf16,
        fp16=opts.use_fp16,
        push_to_hub=False,
        overwrite_output_dir=True,
        seed=opts.seed,
        data_seed=opts.seed,
        # metric_for_best_model='oqo',
        metric_for_best_model='exact_match',
        greater_is_better=True,
        # deepspeed=opts.deepspeed,
    )

    compute_metrics = get_compute_metrics_func(tokenizer, val_dataset)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    test_results = trainer.predict(test_dataset)
    test_preds = test_results.predictions
    test_labels = test_results.label_ids
    
    print('Start evaluating on test set')
    compute_test_metrics = get_compute_metrics_func(tokenizer, test_dataset)
    test_metrics = compute_test_metrics((test_preds, test_labels))
    print("Test metrics:", test_metrics)
        

    shutil.rmtree(output_path)
    trainer.save_model(output_dir=output_path)
    trainer.save_state()
    print('model saved to ' + output_path)


def get_compute_metrics_func(tokenizer, val_dataset):
    decoded_labels = val_dataset.output
    input_texts = val_dataset.input

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # print('')
        # print('preds', preds.shape)
        # print('labels', labels.shape)
        # print('max_input_length', val_dataset.max_input_length)
        # print('max_output_length', val_dataset.max_output_length)
        # print('max_total_length', val_dataset.max_total_length)
        # print('')

        # print('decode pred')
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        assert len(decoded_preds) == len(decoded_labels)

        # print('\n')
        
        # random print prediction vs gold
        random_index = random.randint(0, len(decoded_preds) - 1)

        # print('input:', input_texts[random_index])
        # print('pred:\n', decoded_preds[random_index]) 
        # print('gold:\n', decoded_labels[random_index])

        # print('compute metric')
        metrics = dict()

        exact_match, _ = get_exact_match(decoded_preds, decoded_labels)
        metrics['exact_match'] = round(exact_match, 2)

        # converter.api_calls = 0
        # oqo_score, _ = get_oqo_score(decoded_preds, decoded_labels, converter=converter)
        # print('api calls', converter.api_calls)
        # converter.api_calls = 0
        # try:
        #     converter.save_cache()
        # except Exception:
        #     pass
        # metrics['chrf'] = round(oqo_score['chrf'], 2)
        # metrics['kv_overlap'] = round(oqo_score['kv_overlap'], 2)
        # metrics['xml_overlap'] = round(oqo_score['xml_overlap'], 2)
        # metrics['oqo'] = round(oqo_score['oqo'], 2)

        print(metrics)
        return metrics

    return compute_metrics


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # converter.save_cache()
        exit()
