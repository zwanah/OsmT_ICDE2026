import os
import shutil
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, T5Tokenizer, AutoTokenizer

def check_create_folder(path: str, ask_to_rm_if_exists=False):
    if os.path.exists(path):
        if ask_to_rm_if_exists:
            response = input(
                f"<{path}>: Already exists.\n\nWrite 'del' if you wish to delete other wise press any key: "
            )
            if response.lower() == "del":
                print(f"Deleting: {path}")
                shutil.rmtree(path)

                os.makedirs(path)
    else:
        os.makedirs(path)


def combine_csv_files(csv_paths: list[str], shuffle=False):
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        columns = df.columns

        print(f"Number of records in {path}: {df.shape[0]}")

        if i == 0:
            df_full = df
            columns_base = columns
        else:
            if not np.array_equal(columns, columns_base):
                raise (Exception("Columns do not match"))

            total_records = df_full.shape[0] + df.shape[0]

            df_full = (
                pd.concat([df_full, df]).drop_duplicates(columns).reset_index(drop=True)
            )

            records_dropped = total_records - df_full.shape[0]

            print(f"-> Merged!!, {records_dropped} duplicates were found and dropped\n")

    if shuffle:
        shuffled_indices = np.random.permutation(np.arange(df_full.shape[0]))
        df_full = df_full.iloc[shuffled_indices, :].reset_index(drop=True)

    print(f"A total of {df_full.shape[0]} recrods were loaded")
    return df_full

def get_library(model_type):
    prefix, _, size = model_type.partition('-')

    if prefix == 't5':
        return model_type, T5Tokenizer

    if prefix in ['codet5', 'codet5p']:
        return 'Salesforce/' + model_type, RobertaTokenizer if prefix == 'codet5' else AutoTokenizer
    
    if prefix == 'byt5':
        return 'google/' + model_type, T5Tokenizer

    return None, None


def read_overpass_split(path, prefix='', num=None):
    nls = list()
    with open(path + '.nl') as f:
        for line in f:
            nls.append(prefix + line.strip())

    queries = list()
    with open(path + '.query') as f:
        for line in f:
            queries.append(line.strip())

    assert len(nls) == len(queries)

    if num is not None:
        nls = nls[:num]
        queries = queries[:num]

    return nls, queries

def read_tags_csv(path,num=None):
    df = pd.read_csv(path)
    # tags_all : tag,key,value,ovq_style
    # tags_description : key,value,description,tag,ovq_style
    tags_ovq_style = df['ovq_style'].tolist()
    # to string
    tags_ovq_style = [str(tag) for tag in tags_ovq_style]
    if num is not None:
        tags_ovq_style = tags_ovq_style[:num]
    if 'description' in df.columns:
        tags_description = df['description'].tolist()
        # to string
        tags_description = [str(tag) for tag in tags_description]
        if num is not None:
            tags_description = tags_description[:num]
        return tags_ovq_style,tags_description
    return tags_ovq_style

# add prefix to show the modal
def add_prefix(dataset, prefix):
    # dataset : list
    return [prefix + d for d in dataset]

if __name__ == "__main__":
    # path = 'data/overpass_split/dataset.train'
    # nls, queries = read_overpass_split(path)
    # print(nls[:5])
    # print(queries[:5])
    # print(len(nls), len(queries))
    # print()

    # path = 'data/overpass_split/dataset.dev'
    # nls, queries = read_overpass_split(path)
    # print(nls[:5])
    # print(queries[:5])
    # print(len(nls), len(queries))
    # print()

    # path = 'data/overpass_split/dataset.test'
    # nls, queries = read_overpass_split(path)
    # print(nls[:5])
    # print(queries[:5])
    # print(len(nls), len(queries))
    # print()

    path = '/home/russ/LLM4Geo/OSMT5/osm/tags_all.csv'
    tags_ovq_style = read_tags_csv(path)
    print(tags_ovq_style[:5])
    print(len(tags_ovq_style))
    print()

    path = '/home/russ/LLM4Geo/OSMT5/osm/tags_description.csv'
    tags_ovq_style, tags_description = read_tags_csv(path)
    print(tags_ovq_style[:5])
    print(tags_description[:5])
    print(len(tags_ovq_style), len(tags_description))
    print()