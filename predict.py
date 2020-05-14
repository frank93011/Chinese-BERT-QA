import transformers
import torch
import json
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
from utils import QADataset, collate_fn, preprocess
from transformers import BertForQuestionAnswering
from torch.utils.data import DataLoader
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()
INPUT__PATH = args.test_data_path
OUTPUT_PATH = args.output_path

def clean_word(sentence):
    stop_words = ['##', '[UNK]', '[CLS]']
    for s in stop_words:
        sentence = sentence.replace(s, '')
    if(sentence):
        if(sentence[0]=="《" and sentence[-1] != "》"):
            sentence +=  "》"
        if(sentence[0]!="《" and sentence[-1] == "》"):
            sentence = "《"+sentence
    return sentence

def predict(model, dataloader, device):
    predictions = None
    correct = 0
    total = 0
    total_loss = 0
    predictions = {}
    with torch.no_grad():
        for data in tqdm(dataloader):
            ids = data[5]
            data = [t.to(device) for t in data[:-1]]
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            start_scores, end_scores = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            start_scores = torch.nn.functional.softmax(start_scores)
            end_scores = torch.nn.functional.softmax(end_scores)
            for i in range(len(data[0])):
                id = ids[i]
                all_tokens = tokenizer.convert_ids_to_tokens(tokens_tensors[i])
                start, end = torch.argmax(start_scores[i]), torch.argmax(end_scores[i])
                scorer = start_scores[i]+end_scores[i]
                retry = 1
                while(end - start >= 20 or start > end or (not start and end)):
                    retry += 1
                    _, starts = torch.topk(start_scores[i], retry)
                    _, ends = torch.topk(end_scores[i], retry)
                    if start == 0:
                        start = starts[retry-1]
                    else:
                        if(start_scores[i][start]+end_scores[i][ends[retry-1]] > start_scores[i][starts[retry-1]]+end_scores[i][end]):
                            end = ends[retry-1]
                        else:
                            start = starts[retry-1]
                is_answerable = torch.sigmoid(torch.tensor([scorer[0], (start_scores[i][start] + end_scores[i][end])]))
                if(is_answerable[0] > is_answerable[1]):
                    predictions[id] = ''
                else:
                    temp = ''.join(all_tokens[start:end])
                    predictions[id] = clean_word(temp)

    return predictions

PRETRAINED_MODEL_NAME = "bert-base-chinese" 
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
valid = preprocess(INPUT__PATH, "test")
validset = QADataset(valid, "test", tokenizer=tokenizer)
BATCH_SIZE = 16
validloader = DataLoader(validset, batch_size=BATCH_SIZE, 
                         collate_fn=collate_fn)

PRETRAINED_MODEL_NAME = "bert-base-chinese"
NUM_LABELS = 2
model = BertForQuestionAnswering.from_pretrained(
    PRETRAINED_MODEL_NAME)

model.load_state_dict(torch.load('./save/QA-final.pkl'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.cuda()

predictions = predict(model, validloader,device)
json.dump(predictions, open(OUTPUT_PATH, "w"))