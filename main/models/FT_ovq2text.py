from cgi import test
import os
import argparse
import shutil
import random
import pandas as pd
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

from utils import OverpassDataset
from utils import read_overpass_split, setup_seed, get_comment_queries, read_kv_split

# from evaluation import get_exact_match, get_oqo_score, OverpassXMLConverter
from evaluation import calculate_bleu_scores, calculate_meteor_scores, calculate_rouge_scores

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--data_dir', default='../dataset', type=str)
parser.add_argument('--exp_name', default='debug', type=str) # e30_lr0004
parser.add_argument('--output_dir', default='./outputs', type=str)
parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--model_name', default='Salesforce/codet5-base', type=str,choices=['Salesforce/codet5-small', 'Salesforce/codet5-base', 'Salesforce/codet5-large',
                                                                                        'Salesforce/codet5p-220m', 'Salesforce/codet5p-770m',
                                                                                        'google/byt5-small','google/byt5-base', 'google/byt5-large',
                                                                                        'facebook/bart-base'])

parser.add_argument('--ckpt_path', default=None, type=str)

parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--batch_size', default=16, type=int) # 32 Salesforce/codet5p-220m 
parser.add_argument('--eval_batch_size', default=64, type=int)
parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
parser.add_argument('--max_length', default=512, type=int)  # max_length=600 for byt5?
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.1, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--learning_rate', default=4e-4, type=float)
parser.add_argument('--lr_scheduler_type', default='linear', type=str)
parser.add_argument('--converter_url', default="http://localhost:12346/api/convert", type=str)

parser.add_argument('--use_prefix', default=False, type=bool)
parser.add_argument('--use_comments_task', default=False, type=bool) # if use bs/2
parser.add_argument('--use_key_value', default=False, type=bool)
parser.add_argument('--use_primary_kv', default=False, type=bool)
parser.add_argument('--comment_max_count', default=10, type=int)
parser.add_argument('--comment_min_content_ratio', default=0.80, type=float)

parser.add_argument('--reverse', default=True, type=bool)

parser.add_argument('--use_fp16', default=True, type=bool)
# parser.add_argument("--deepspeed", type=str, default='configs/ds_config_zero2.json', help="Path to deepspeed config file.")

opts = parser.parse_args()
setup_seed(opts.seed)

if opts.reverse:
    group = 'ovq2text'
else: 
    group = 'text2ovq'
print('config', opts)

# converter = OverpassXMLConverter(url=opts.converter_url, save_frequency=-1)


def main():
    if opts.model_name == 'google/byt5-small' or opts.model_name == 'google/byt5-base' or opts.model_name == 'google/byt5-large':
        opts.max_length = 600
        opts.use_fp16 = False
        if opts.model_name == 'google/byt5-small':
            eval_steps = 400
        else:
            eval_steps = 800
    elif opts.model_name == 'facebook/bart-base':
        opts.use_fp16 = False
        eval_steps = 200
    else:
        eval_steps = 200
        
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
        # prefix_nl = '<NL> '
        prefix_nl = ''
        prefix_query = '<OVQ> '
        # prefix_query = ''
    else:
        prefix_nl = ''
        prefix_query = ''
        
    train_nl, train_query = read_overpass_split(opts.data_dir + "/dataset.train", prefix_nl, prefix_query)
    val_nl, val_query = read_overpass_split(opts.data_dir + "/dataset.dev", prefix_nl, prefix_query)
    test_nl, test_query = read_overpass_split(opts.data_dir + "/dataset.test", prefix_nl, prefix_query)

    # if opts.use_key_value:  
    #     if opts.use_prefix:
    #         prefix_tag = '<Tag> '
    #     else:
    #         prefix_tag = ''
    #     train_kv_texts = read_kv_split(opts.data_dir + "/dataset.train", primary_kv=opts.use_primary_kv, prefix=prefix_tag)
    #     val_kv_texts = read_kv_split(opts.data_dir + "/dataset.dev", primary_kv=opts.use_primary_kv, prefix=prefix_tag)
    #     test_kv_texts = read_kv_split(opts.data_dir + "/dataset.test", primary_kv=opts.use_primary_kv, prefix=prefix_tag)
    #     # add key value to the last token of the input
    #     train_texts = [f"{t} {kv}" for t, kv in zip(train_texts, train_kv_texts)]
    #     val_texts = [f"{t} {kv}" for t, kv in zip(val_texts, val_kv_texts)]
    #     test_texts = [f"{t} {kv}" for t, kv in zip(test_texts, test_kv_texts)]
    
    comments_dataset = None
    if opts.use_comments_task:
        print('use comments task')
        comments_nl, comments_query = get_comment_queries(opts.data_dir + '/comments.jsonl',
                                                              comment_max_count=opts.comment_max_count,
                                                              comment_min_content_ratio=opts.comment_min_content_ratio)
        comments_dataset = OverpassDataset(comments_nl, comments_query, tokenizer, 'train',reverse=opts.reverse)

    train_dataset = OverpassDataset(train_nl, train_query, tokenizer, 'train', comments_dataset=comments_dataset,reverse=opts.reverse)
    val_dataset = OverpassDataset(val_nl, val_query, tokenizer, 'val',reverse=opts.reverse)
    test_dataset = OverpassDataset(test_nl, test_query, tokenizer, 'test',reverse=opts.reverse)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest')

    output_name = f"{opts.model_name.split('/')[1]}_{opts.exp_name}"
    output_path = os.path.join(opts.output_dir, group, output_name)
    result_path = os.path.join(opts.result_dir, group, output_name)
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
        # bf16=True,
        fp16=opts.use_fp16,
        push_to_hub=False,
        overwrite_output_dir=True,
        seed=opts.seed,
        data_seed=opts.seed,
        # metric_for_best_model='oqo',
        metric_for_best_model='meteor',
        greater_is_better=True,
        # deepspeed="configs/ds_config_zero2.json",
        run_name=output_name,
    )

    compute_metrics = get_translate_metric(tokenizer, val_dataset)

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
    compute_test_metrics = get_translate_metric(tokenizer, test_dataset, all_metrics=True, result_path=result_path)
    test_metrics = compute_test_metrics((test_preds, test_labels))
    print("Test metrics:", test_metrics)
        

    shutil.rmtree(output_path)
    trainer.save_model(output_dir=output_path)
    trainer.save_state()
    print('model saved to ' + output_path)


def get_translate_metric(tokenizer, val_dataset,all_metrics=False,result_path=None):
    decoded_labels = val_dataset.output
    input_texts = val_dataset.input
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print('')
        print('preds', preds.shape)
        print('labels', labels.shape)
        print('max_input_length', val_dataset.max_input_length)
        print('max_output_length', val_dataset.max_output_length)
        print('max_total_length', val_dataset.max_total_length)
        print('')

        print('decode pred')
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        assert len(decoded_preds) == len(decoded_labels)

        # Randomly print prediction vs gold
        random_index = random.randint(0, len(decoded_preds) - 1)
        print('\nExample:')
        print('input:', input_texts[random_index])
        print('pred:\n', decoded_preds[random_index]) 
        print('gold:\n', decoded_labels[random_index])

        print('compute metric')
        metrics = dict()
        
        meteor = calculate_meteor_scores(decoded_labels, decoded_preds)
        if all_metrics:
            bleu1 = calculate_bleu_scores(decoded_labels, decoded_preds, max_order=1)
            bleu2 = calculate_bleu_scores(decoded_labels, decoded_preds, max_order=2)
            bleu4 = calculate_bleu_scores(decoded_labels, decoded_preds, max_order=4)
            rouge1 = calculate_rouge_scores(decoded_labels, decoded_preds, 'rouge1')
            rouge2 = calculate_rouge_scores(decoded_labels, decoded_preds, 'rouge2')
            rougeL = calculate_rouge_scores(decoded_labels, decoded_preds, 'rougeL')
        
            metrics = {
                'bleu1': bleu1,
                'bleu2': bleu2,
                'bleu4': bleu4,
                'rouge1': rouge1,
                'rouge2': rouge2,
                'rougeL': rougeL,
                'meteor': meteor
            }
            # save test results
            csv_file_path = os.path.join(result_path, 'test_results.csv')
            # Check if the directory exists, if not create it
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            results_df = pd.DataFrame({
                'Input': input_texts,
                'Prediction': decoded_preds,
                'Gold Label': decoded_labels
            })
            results_df.to_csv(csv_file_path, index=False)
            print(f'Results saved to {csv_file_path}')
            
            # save metrics
            metrics_file_path = os.path.join(result_path, 'metrics.txt')
            with open(metrics_file_path, 'w') as f:
                f.write(f'BLEU1: {bleu1}\n')
                f.write(f'BLEU2: {bleu2}\n')
                f.write(f'BLEU4: {bleu4}\n')
                f.write(f'ROUGE1: {rouge1}\n')
                f.write(f'ROUGE2: {rouge2}\n')
                f.write(f'ROUGEL: {rougeL}\n')
                f.write(f'METEOR: {meteor}\n')
        else:
            metrics = {
                'meteor': meteor
            }

        print('\nMetrics:', metrics)
        return metrics

    return compute_metrics



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # converter.save_cache()
        exit()
