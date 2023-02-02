import torch
from argparse import ArgumentParser
from model import UnscramblingModule
from transformers import AutoTokenizer


parser = ArgumentParser()

parser.add_argument('--tokenizer', type=str, default='m3hrdadfi/bert2bert-fa-wiki-summary', help='tokenizer')
parser.add_argument('--model', type=str, default='m3hrdadfi/bert2bert-fa-wiki-summary', help='model')
parser.add_argument('--model_checkpoint', type=str, default='model.pth', help='dataset directory')
parser.add_argument('--input_words', type=str, default='input_words.txt')

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UnscramblingModule(args.model)
model.load_state_dict(torch.load(args.model_checkpoint))
model.to(device)


def convert(text):
    text_encoding = tokenizer(
      text,
      padding=True,
      return_tensors='pt'
    )
    text_encoding = {k: v.to(device) for k, v in text_encoding.items()}

    generated_ids = model.transformer.generate(
      input_ids=text_encoding['input_ids'],
      attention_mask=text_encoding['attention_mask'],
      early_stopping=True
    )

    preds = [
           tokenizer.decode(gen_id, skip_special_tokens=True,
                            clean_up_tokenization_spaces=True) for gen_id in generated_ids
    ]

    return "".join(preds)


with open(args.informal_texts, 'r') as file:
    words_list = []
    for line in file:
        words_list.append(line)

res = []
for words in words_list:
    res.append(convert(words))

for r in res:
    print(r)
