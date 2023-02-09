import torch
from argparse import ArgumentParser
from model import UnscramblingModule
from transformers import AutoTokenizer
from utils import read_dataset
from dataset import UnscramblingDataset
from torch.utils.data import DataLoader
from utils import collate_fn
import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

parser = ArgumentParser()

parser.add_argument('--tokenizer', type=str, default='m3hrdadfi/bert2bert-fa-wiki-summary', help='tokenizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--model', type=str, default='m3hrdadfi/bert2bert-fa-wiki-summary', help='model')
parser.add_argument('--model_checkpoint', type=str, default='model.pth', help='checkpoint directory')
parser.add_argument('--last_5_percent', type=bool, default=False, help='use last 5 percent of data to test model.')
parser.add_argument('--validation_path', type=str, default='val_informals.csv')
parser.add_argument('--column_names', type=str, default='Shuffled Original', help='column names separated with space. first shuffled and second orginal.')

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UnscramblingModule(args.model)
model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
model.to(device)
model.eval()


rouge = Rouge()


def convert_compute(input_ids, attention_mask, labels):
    generated_ids = model.transformer.generate(
        input_ids=input_ids,
        attention_mask=attention_mask, max_new_tokens=150
    )

    pred_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    labels[labels == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}

    rouge_output = rouge.get_scores(pred_str, label_str)
    for row in rouge_output:
        scores['rouge-1'] += row['rouge-1']['f']
        scores['rouge-2'] += row['rouge-2']['f']
        scores['rouge-l'] += row['rouge-l']['f']

    scores['rouge-1'] = scores['rouge-1'] / len(rouge_output)
    scores['rouge-2'] = scores['rouge-2'] / len(rouge_output)
    scores['rouge-l'] = scores['rouge-l'] / len(rouge_output)

    return input_str, pred_str, label_str, scores


_, val_sentences = read_dataset(args.validation_path, args.column_names.split(), split=not args.last_5_percent, last_5_percent=args.last_5_percent)

val_dataset = UnscramblingDataset(val_sentences)
val_dataloader = DataLoader(val_dataset, collate_fn=lambda data: collate_fn(data, tokenizer),
                            batch_size=args.batch_size)

print(f'you are using {device}')

d = {'words': [], 'prediction': [], 'ground_truth': []}
scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
i = 0
with tqdm(val_dataloader, unit="batch") as tepoch:
    for batch in tepoch:
        tepoch.set_description(f"Validation")

        batch = {k: v.to(device) for k, v in batch.items() if k != 'token_type_ids'}
        inputs_str, preds_str, labels_str, output_scores = convert_compute(**batch)

        scores = {k: scores[k] + v for k, v in output_scores.items()}
        i += 1

        d['words'] = d['words'] + inputs_str
        d['prediction'] = d['prediction'] + preds_str
        d['ground_truth'] = d['ground_truth'] + labels_str
        tepoch.set_postfix(rouge1=output_scores['rouge-1'], rouge2=output_scores['rouge-2'],
                           rougel=output_scores['rouge-l'])

scores = {k: v / i for k, v in scores.items()}
print(scores)
df = pd.DataFrame(d)
df.to_csv('validation.csv')
