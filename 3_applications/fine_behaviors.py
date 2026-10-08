# classify_text(des, categories): given a description, fit the desription into a category
# categories: dict. keys are cates, values are lists which contain keywords or phrases
# update_categories(categories, api_key, unknow_des): given an unknown description update the categories
# categories = {
    # 'cat1' : [],
    # 'cat2' : [],
    # ...
# }

import re
import json
import numpy as np
import pandas as pd
import os
import argparse
from config import resolve_groups
from llm_utils import request_json



def validate_categories(categories):
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Categories must be a nonempty object.")
    for name, keywords in categories.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(keywords, list):
            raise ValueError("Each category must have a name and a list of keywords.")
        if name != "Empty" and not all(isinstance(keyword, str) and keyword.strip() for keyword in keywords):
            raise ValueError("Keywords must be nonempty strings.")
    return categories


def gpt_request(prompt, *, max_attempts=3, keyword_source=None, existing_categories=None):
    def validate(result):
        validate_categories(result.get("categories"))
        keyword = result.get("keyword")
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("Model response must contain a nonempty keyword.")
        if keyword_source is not None and keyword not in keyword_source:
            raise ValueError("The keyword must occur verbatim in the description.")
        if keyword not in [value for values in result["categories"].values() for value in values]:
            raise ValueError("The extracted keyword must be added to a category.")
        if existing_categories is not None:
            for name, keywords in existing_categories.items():
                if name not in result["categories"] or not set(keywords).issubset(result["categories"][name]):
                    raise ValueError("Category discovery must preserve existing names and keywords.")
            result["categories"] = {
                **{name: result["categories"][name] for name in existing_categories},
                **result["categories"],
            }
        return result
    return request_json(
        client, prompt,
        "You are an expert on mouse behaviors. Design unique keywords or short phrases for each behavior. Output a JSON object.",
        validate, max_attempts=max_attempts,
    )


def update_categories(prompt, categories, unknow_des, *, max_attempts=3):
    data_prompt = f"\nUnknown description: {unknow_des}\nCategories: {categories}\n"
    result = gpt_request(prompt + data_prompt, max_attempts=max_attempts, keyword_source=unknow_des, existing_categories=categories)
    return result["categories"]


def classify_text(text, categories):
    text = text.lower()
    cats = set()
    for category, keywords in categories.items():
        if category == 'Empty': continue
        for keyword in keywords:
            if keyword.lower() in text:
                cats.add(category)
                return cats
    if len(cats) != 0:
        return cats
    else:
        return 'Unknown'

def classify_texts(texts, categories, part_name):
    validate_categories(categories)
    labels = np.full(len(texts), -1, dtype=np.int64)
    count = {"Unknown": 0, "Empty": 0, **dict.fromkeys(categories, 0)}
    unknown_descriptions = []
    for index, text in enumerate(texts):
        if isinstance(text, dict):
            if part_name == "All":
                text = " ".join(str(value) for key, value in text.items() if key != "Keywords")
            else:
                text = text.get(part_name, "")
        if not isinstance(text, str):
            raise ValueError(f"Description {index} must contain text.")
        if not text.strip():
            count["Empty"] += 1
            continue
        categories_found = classify_text(text, categories)
        if categories_found == "Unknown":
            count["Unknown"] += 1
            unknown_descriptions.append(text)
        else:
            category = next(iter(categories_found))
            count[category] += 1
            labels[index] = list(categories).index(category)
    return labels, count, unknown_descriptions


def load_categories(path):
    with open(path, 'r') as file:
        categories = json.load(file)
    return validate_categories(categories)


def get_labels_category(prompt, categories, texts, part_name, *, max_attempts=3, max_category_updates=100, output_path=None):
    if max_category_updates < 1:
        raise ValueError("max_category_updates must be positive.")
    labels, count, unknown = classify_texts(texts, categories, part_name)
    for _ in range(max_category_updates):
        if not unknown:
            break
        updated = update_categories(prompt, categories, unknown[0], max_attempts=max_attempts)
        next_labels, next_count, next_unknown = classify_texts(texts, updated, part_name)
        if next_count["Unknown"] >= count["Unknown"]:
            raise RuntimeError("Category update did not reduce unclassified descriptions; stopping requests.")
        categories, labels, count, unknown = updated, next_labels, next_count, next_unknown
    if unknown:
        raise RuntimeError(f"Category discovery stopped after {max_category_updates} updates with {len(unknown)} unclassified descriptions.")
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(categories, handle, indent=4)
    return labels, count, categories


def main(args):
    global output_cate_json_path
    ofolder = args.ofolder
    os.makedirs(ofolder, exist_ok=True)
    output_cate_json_path = f'{args.ofolder}/{args.output_json}'
    group_config = args.group_config
    category_path = args.category
    workspace = args.workspace
    prompt_path = args.prompt
    part_name = args.part_name
    model_name = args.model_name
    date = args.date
    
    # Initialize an empty list for the groups
    groups = resolve_groups(group_config)
    print(f"Selected group configurations: {group_config}")
    print(groups)
    
    # load categories
    categories = load_categories(category_path)
    
                
    for n, group in enumerate(groups):
                
        # load descriptions
        h5_file_path = f'{workspace}/{model_name}/{group}.h5'
        with pd.HDFStore(h5_file_path) as store:
            texts = store.get('descriptions')['description']
            ids = store.get('descriptions').id
        
        with open(prompt_path, 'r') as file:
            prompt = file.read()
        labels, count, categories = get_labels_category(prompt, categories, texts, part_name, max_attempts=args.max_attempts, max_category_updates=args.max_category_updates, output_path=output_cate_json_path)
    
    
    data = []
    header = []

    header.append('group')
    
    for n, group in enumerate(groups):
        # load descriptions
        h5_file_path = f'{workspace}/{model_name}/{group}.h5'
        with pd.HDFStore(h5_file_path) as store:
            texts = store.get('descriptions')['description']
            ids = store.get('descriptions').id
        
        with open(prompt_path, 'r') as file:
            prompt = file.read()
        labels, count, categories = get_labels_category(prompt, categories, texts, part_name, max_attempts=args.max_attempts, max_category_updates=args.max_category_updates, output_path=output_cate_json_path)
    
    
        behavior_category_keywords = pd.DataFrame(categories.keys())
        labels_df = pd.DataFrame(labels, ids)
        
        
        with pd.HDFStore(h5_file_path) as store:
            store.put(f'category_{part_name}_keyword_{date}', behavior_category_keywords)
            store.put(f'label_{part_name}_keyword_{date}', labels_df)

        group_data = [group]
        for i, (k, v) in enumerate(count.items()):
            if i == 0: continue
            if n == 0:
                header.append(k)
            group_data.append(v)

        if n == 0:
            data.append(header)
        data.append(group_data)

    df = pd.DataFrame(data)
    print(df)
    filename = f'{ofolder}/{model_name}_{group_config[0]}_{part_name}_keyword.xlsx'
    df.to_excel(filename, index=False)

    print(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--part_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--group_config', type=str, nargs='+', required=True)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--ofolder', type=str, required=True)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_category_updates", type=int, default=100)
    args = parser.parse_args()


    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        parser.error("Set OPENAI_API_KEY in your environment before running this script.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, max_retries=0, timeout=60)
    
    
    main(args)
