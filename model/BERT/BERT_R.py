import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm 
import os
import datetime


num_classes = 78 # 분류하고자 하는 클래스 개수
learning_rate = 1.23e-5 #lr 값 조정

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("\n\nTrain Start Time =", current_time)
print("Train shuffle= True, Eval shuffle= True")
print(f"num_classes = {num_classes}")
print(f"lr = {learning_rate}\n\n")

train_df = pd.read_csv('./data/BERT_train.csv')
test_df = pd.read_csv('./data/BERT_test.csv')

#결측치 제거
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_df['sleep_score'] = pd.cut(train_df['sleep_score'], bins=num_classes, labels=False)
test_df['sleep_score'] = pd.cut(test_df['sleep_score'], bins=num_classes, labels=False)


class NsmcDataset(Dataset):
    ''' Naver Sentiment Movie Corpus Dataset '''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]

        return text, label


nsmc_train_dataset = NsmcDataset(train_df)


####
train_loader = DataLoader(nsmc_train_dataset, batch_size=16, shuffle=True, num_workers=32)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

config = BertConfig.from_pretrained('bert-base-multilingual-cased')
config.num_hidden_layers = 10
config.num_labels = 78

print("<<config info>>\n", config)

device = torch.device("cuda:0")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', config=config)


model.to(device) #cpu -> gpu로 모델 이동

optimizer = Adam(model.parameters(), lr=learning_rate) #learning_rate 위에서 조정.

itr = 1
p_itr = 20
epochs = 10
total_loss = 0
total_len = 0
total_correct = 0


model.train()

for epoch in range(epochs):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for text, label in progress_bar:
        optimizer.zero_grad()

        encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
        padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
        sample = torch.as_tensor(padded_list)
        sample, label = sample.to(device), label.to(device)
        labels = torch.as_tensor(label)
        outputs = model(sample, labels=labels)
        loss, logits = outputs

        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if itr % p_itr == 0:
            progress_bar.set_postfix({"Train Loss": total_loss/p_itr, "Accuracy": total_correct/total_len})
            total_loss = 0
            total_len = 0
            total_correct = 0

        itr += 1
        
from sklearn.metrics import classification_report, precision_recall_fscore_support

model.eval()

nsmc_eval_dataset = NsmcDataset(test_df)

eval_loader = DataLoader(nsmc_eval_dataset, batch_size=8, shuffle=True, num_workers=16)

total_loss = 0
total_len = 0
total_correct = 0

y_true = []
y_pred = []

for text, label in tqdm(eval_loader, desc='Evaluation'):
    encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
    padded_list =  [e + [0] * (512-len(e)) for e in encoded_list]
    sample = torch.as_tensor(padded_list)
    sample, label = sample.to(device), label.to(device)
    labels = torch.as_tensor(label)
    outputs = model(sample, labels=labels)
    _, logits = outputs

    pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
    correct = pred.eq(labels)
    total_correct += correct.sum().item()
    total_len += len(labels)
    total_loss += loss.item()
    
    y_true += label.tolist()
    y_pred += pred.tolist()


print(classification_report(y_true, y_pred, zero_division=1))

accuracy = total_correct / total_len
precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)
loss = total_loss / len(eval_loader)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')
print(f'Loss: {loss:.4f}')

report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f'./user_classification_report/{current_time}_ly{config.num_hidden_layers}lr{learning_rate}_ep{epochs}.csv', index=True)


if os.path.isfile('result.csv'):
    df = pd.read_csv('result.csv')
else:
    df = pd.DataFrame(columns=['Lr','Epoch','Layer','Accuracy', 'Precision', 'Recall', 'F1 Score', 'Loss'])


result = [[learning_rate, epochs, config.num_hidden_layers, accuracy, precision, recall, f1_score, loss]]
new_df = pd.DataFrame(result, columns=['Lr','Epoch','Layer','Accuracy', 'Precision', 'Recall', 'F1 Score', 'Loss'])
df = df.append(new_df, ignore_index=True)


df.to_csv('result.csv', index=False)

now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("\nTrain End Time =", current_time)
