from torch.utils.data import Dataset
import json
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


def preprocess(data_path, mode):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    data = data['data']
    data_list = []
    if(mode !="test"):
        for d in data:
            para = d['paragraphs']
            for p in para:
                context = p['context']
                for q in p['qas']:
                    data_list.append({'context': context, 'id':q['id'], 'question':q['question'], 'answers':q['answers'], 'answerable':q['answerable']})
    else:
        for d in data:
            para = d['paragraphs']
            for p in para:
                context = p['context']
                for q in p['qas']:
                    data_list.append({'context': context, 'id':q['id'], 'question':q['question']})
    return pd.DataFrame(data_list)

class QADataset(Dataset):
    def __init__(self, df, mode, tokenizer, MAX_LENGTH=512):
        self.df = df
        self.mode = mode
        self.len = len(df)
        self.tokenizer = tokenizer
        self.max_len = MAX_LENGTH
    
    # 重新計算ans token在context tokens 的位置
    def token_start_end(self, tokenizedTokens, target):
        if not target:
            return 0,0
        target_len = len(target)
        for i, t in enumerate(tokenizedTokens):
            if t == target[0] and i <= len(tokenizedTokens)-target_len:
                if(target == tokenizedTokens[i:i+target_len]):
                    return (i, i+target_len)
        return 0,0
    
    def __getitem__(self, idx):
        ans_token = ""
        if self.mode == "test":
            contexts, id, question  = self.df.iloc[idx, :3].values
            label_tensor = None
        else:
            contexts, id, question, answers, answerable  = self.df.iloc[idx, :5].values
            ans_token = self.tokenizer.tokenize(answers[0]['text'])
            ans_start = answers[0]['answer_start']
            if(ans_start >= self.max_len-3-len(question)):
                contexts_len = self.max_len-3-len(question)
                contexts = contexts[(ans_start - contexts_len//2):]
        
        contexts = contexts[:self.max_len-3-len(question)]
   
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(contexts)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        # 找到anwer 在 contexts 中的 start end
        start, end = self.token_start_end(word_pieces, ans_token)
        start_label = torch.tensor(start)
        end_label = torch.tensor(end)

        tokens_b = self.tokenizer.tokenize(question)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        
        # convert token to indexs
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, start_label, end_label, id)
    def __len__(self):
        return self.len
    
def collate_fn(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    start_label = torch.tensor([s[2] for s in samples])
    end_label = torch.tensor([s[3] for s in samples])
    questions_ids = [s[4] for s in samples]
    
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, start_label, end_label, questions_ids
