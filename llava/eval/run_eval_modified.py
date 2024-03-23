import argparse
import json
import collections
import pandas as pd    
from nltk.translate.bleu_score import sentence_bleu
from eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate
from eval_metrics.glossary import *

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file')
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeroshot.jsonl", help='path to prediction file')
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def evaluate(gt, pred):    
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    recall_scores = collections.defaultdict(list)
    precision_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    open_hit_scores = collections.defaultdict(list)
    closed_scores = collections.defaultdict(list)

    for gt_item, pred_item in zip(gt, pred):
        try:
            gt_results = gt_item['conversations']
        except KeyError:
            gt_results = gt_item['conversatons']
        gt_value = gt_results[1]['value'].lower()
        pred_value = pred_item['text'].lower()
        print(f"Question: {gt_results[0]['value'].lower()} || ground truth: {gt_value} || predicted: {pred_value}" )
        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        # For closed-ended questions (Yes/No)
        if gt_value in ["yes", "no"]:
            closed_scores['q_id'].append(pred_item['question_id'])
            if gt_value in pred_value:
                closed_scores['hit'].append(1)
            else:
                closed_scores['hit'].append(0)
        
        # Calculate scores for open-ended questions
        else:
            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores['f1'].append(f1_score)
            precision_scores['precision'].append(precision)
            recall_scores['recall'].append(recall)
            b_score = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split())
            b_score_1 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(1, 0, 0, 0))
            b_score_2 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 1, 0, 0))
            b_score_3 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 0, 1, 0))
            
            bleu_scores['q_id'].append(pred_item['question_id'])
            bleu_scores['bleu_score'].append(b_score)
            bleu_scores['bleu_score_1'].append(b_score_1)
            bleu_scores['bleu_score_2'].append(b_score_2)
            bleu_scores['bleu_score_3'].append(b_score_3)

    # Compute aggregated scores
    closed_score = sum(closed_scores['hit']) / len(closed_scores['hit']) if closed_scores['hit'] else 0
    exact_score = sum(exact_scores['hit']) / len(exact_scores['hit']) if exact_scores['hit'] else 0
    f1_score = sum(f1_scores['f1']) / len(f1_scores['f1']) if f1_scores['f1'] else 0
    precision = sum(precision_scores['precision']) / len(precision_scores['precision']) if precision_scores['precision'] else 0
    recall = sum(recall_scores['recall']) / len(recall_scores['recall']) if recall_scores['recall'] else 0
    bleu_score = sum(bleu_scores['bleu_score']) / len(bleu_scores['bleu_score']) if bleu_scores['bleu_score'] else 0
    bleu_score_1 = sum(bleu_scores['bleu_score_1']) / len(bleu_scores['bleu_score_1']) if bleu_scores['bleu_score_1'] else 0
    bleu_score_2 = sum(bleu_scores['bleu_score_2']) / len(bleu_scores['bleu_score_2']) if bleu_scores['bleu_score_2'] else 0
    bleu_score_3 = sum(bleu_scores['bleu_score_3']) / len(bleu_scores['bleu_score_3']) if bleu_scores['bleu_score_3'] else 0


    # Print results
    results_table = [
            ['exact match score', exact_score*100], 
            ['f1 score', f1_score*100], 
            ['precision', precision*100], 
            ['recall', recall*100], 
            ['bleu_score', bleu_score*100], 
            ['bleu_score_1', bleu_score_1*100], 
            ['bleu_score_2', bleu_score_2*100], 
            ['bleu_score_3', bleu_score_3*100], 
            ['yes/no accuracy', closed_score*100]
    ]
    print(tabulate(results_table, headers=['Metric', 'Performance(%)']))

if __name__ == '__main__':
    args = parse_option()

    gt = json.load(open(args.gt, 'r'))
    pred = load_jsonl(args.pred)

    results = evaluate(gt, pred)
    print(results)