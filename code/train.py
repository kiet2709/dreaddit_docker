# ================================
# 1. IMPORT
# ================================
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import os
import time

# ================================
# 2. CẤU HÌNH (ĐÃ TỐI ƯU CHO GPU 4GB)
# ================================
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
DATA_DIR = "/app/data"
BATCH_SIZE = 16              # GIẢM XUỐNG 16 (từ 64)
EPOCHS = 3
LR = 1e-3
MAX_LEN = 256                # GIẢM XUỐNG 256 (từ 512)
VAL_SPLIT = 0.2
HIDDEN_SIZE = 256
NUM_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.3
NUM_WORKERS = 2              # Giảm để tránh CPU bottleneck
GRAD_ACCUM_STEPS = 4         # Tích lũy gradient → hiệu quả như batch=64

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*20} THIẾT BỊ {'='*20}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    torch.cuda.empty_cache()  # Xóa cache
print(f"{'='*50}\n")

# ================================
# 3. LOAD DREADDIT CSV
# ================================
print(f"{'='*20} LOAD DREADDIT CSV {'='*20}")
train_file = os.path.join(DATA_DIR, "dreaddit_train.csv")
test_file = os.path.join(DATA_DIR, "dreaddit_test.csv")

if not os.path.exists(train_file) or not os.path.exists(test_file):
    raise FileNotFoundError("Không tìm thấy dreaddit_train.csv hoặc dreaddit_test.csv!")

train_df = pd.read_csv(train_file)[['text', 'label']].dropna()
test_df = pd.read_csv(test_file)[['text', 'label']].dropna()

print(f"Original Train: {len(train_df)} | Test: {len(test_df)}")

train_df, val_df = train_test_split(
    train_df, test_size=VAL_SPLIT, stratify=train_df['label'], random_state=42
)

print(f"→ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train labels: {dict(train_df['label'].value_counts())}")
print(f"{'='*50}\n")

# ================================
# 4. TOKENIZER & TINYBERT (FROZEN)
# ================================
print(f"{'='*20} TẢI TINYBERT {'='*20}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)
bert_model.to(device)
bert_model.eval()
for param in bert_model.parameters():
    param.requires_grad = False
print("TinyBERT loaded (frozen)")
print(f"{'='*50}\n")

# ================================
# 5. DATASET
# ================================
class StressDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ================================
# 6. DATALOADER (pin_memory=False để tiết kiệm VRAM)
# ================================
train_dataset = StressDataset(train_df['text'].tolist(), train_df['label'].tolist())
val_dataset   = StressDataset(val_df['text'].tolist(),   val_df['label'].tolist())
test_dataset  = StressDataset(test_df['text'].tolist(),  test_df['label'].tolist())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
print(f"Batches → Train: {len(train_loader)} | Effective batch size: {effective_batch}\n")
print(f"{'='*50}\n")

# ================================
# 7. MODEL
# ================================
class BERT_LSTM(nn.Module):
    def __init__(self, bert_model, hidden_size, num_layers, bidirectional, dropout, num_labels=2):
        super().__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(
            input_size=312,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        lstm_dim = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeds = bert_out.last_hidden_state
        lstm_out, _ = self.lstm(embeds)
        pooled = lstm_out[:, -1, :]
        logits = self.classifier(pooled)
        return logits

model = BERT_LSTM(bert_model, HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ================================
# 8. HÀM ĐÁNH GIÁ
# ================================
def evaluate(loader, desc):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels, total_loss / len(loader)

# ================================
# 9. HUẤN LUYỆN VỚI GRADIENT ACCUMULATION
# ================================
print(f"{'='*20} BẮT ĐẦU HUẤN LUYỆN {'='*20}")
best_val_f1 = 0.0
optimizer.zero_grad()

for epoch in range(EPOCHS):
    print(f"\nEPOCH {epoch+1}/{EPOCHS}")
    print("-" * 60)

    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader, desc="Training  ")
    for i, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels) / GRAD_ACCUM_STEPS
        loss.backward()

        epoch_loss += loss.item() * GRAD_ACCUM_STEPS

        if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        loop.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

    print(f"→ Train Loss: {epoch_loss/len(train_loader):.4f}")

    # VALIDATION
    val_preds, val_labels, val_loss = evaluate(val_loader, "Validating")
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='binary')
    print(f"→ Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        os.makedirs("/app/model_best", exist_ok=True)
        torch.save(model.state_dict(), "/app/model_best/pytorch_model.bin")
        tokenizer.save_pretrained("/app/model_best")
        print(f"→ Model tốt nhất (F1: {val_f1:.4f})")

print(f"\nHUẤN LUYỆN HOÀN TẤT! Best Val F1: {best_val_f1:.4f}")
print(f"{'='*50}\n")

# ================================
# 10. TEST
# ================================
print(f"{'='*20} TEST {'='*20}")
test_preds, test_labels, _ = evaluate(test_loader, "Testing   ")
test_acc = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average='binary')

print(f"\nAccuracy : {test_acc:.4f}")
print(f"F1-Score : {test_f1:.4f}")
print(classification_report(test_labels, test_preds, target_names=['Not Stress', 'Stress']))
print(f"{'='*50}\n")

# ================================
# 11. LƯU MODEL
# ================================
os.makedirs("/app/model_final", exist_ok=True)
torch.save(model.state_dict(), "/app/model_final/pytorch_model.bin")
tokenizer.save_pretrained("/app/model_final")
print("Model lưu tại /app/model_best và /app/model_final")