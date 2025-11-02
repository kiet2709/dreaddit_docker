# ================================
# 1. IMPORT
# ================================
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import os
import time

# ================================
# 2. CẤU HÌNH
# ================================
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
DATA_DIR = "/app/data"  # Docker: /app/data | Colab: "data"
BATCH_SIZE = 32
EPOCHS = 3
LR = 3e-5
MAX_LEN = 256
VAL_SPLIT = 0.2  # 20% cho validation

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*20} THIẾT BỊ {'='*20}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*50}\n")

# ================================
# 3. LOAD & CHIA DỮ LIỆU
# ================================
print(f"{'='*20} LOAD & CHIA DỮ LIỆU {'='*20}")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

# Kết hợp text
train_df['text'] = train_df['Body_Title']
test_df['text'] = test_df['Body_Title']

print(f"Original Train: {len(train_df)} | Test: {len(test_df)}")

# CHIA TRAIN → TRAIN (80%) + VAL (20%)
train_df, val_df = train_test_split(
    train_df, test_size=VAL_SPLIT, stratify=train_df['label'], random_state=42
)

print(f"→ Sau chia: Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train labels: {train_df['label'].value_counts().to_dict()}")
print(f"Val labels:   {val_df['label'].value_counts().to_dict()}")
print(f"Test labels:  {test_df['label'].value_counts().to_dict()}\n")
print(f"{'='*50}\n")

# ================================
# 4. DATASET CLASS
# ================================
class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
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
# 5. LOAD MODEL & TOKENIZER
# ================================
print(f"{'='*20} TẢI TINYBERT {'='*20}")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)
print(f"Loaded in {time.time() - start:.2f}s\n")
print(f"{'='*50}\n")

# ================================
# 6. TẠO DATALOADER
# ================================
train_dataset = StressDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
val_dataset   = StressDataset(val_df['text'].tolist(),   val_df['label'].tolist(),   tokenizer)
test_dataset  = StressDataset(test_df['text'].tolist(),  test_df['label'].tolist(),  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Batches → Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}\n")
print(f"{'='*50}\n")

# ================================
# 7. OPTIMIZER
# ================================
optimizer = AdamW(model.parameters(), lr=LR)

# ================================
# 8. HÀM ĐÁNH GIÁ (DÙNG CHUNG CHO VAL & TEST)
# ================================
def evaluate(loader, desc):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        loop = tqdm(loader, desc=desc, leave=False)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    return all_preds, all_labels, avg_loss

# ================================
# 9. HUẤN LUYỆN + VALIDATION MỖI EPOCH
# ================================
print(f"{'='*20} BẮT ĐẦU HUẤN LUYỆN {'='*20}")
model.train()
best_val_f1 = 0.0
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    print(f"\nEPOCH {epoch+1}/{EPOCHS}")
    print("-" * 60)
    
    # --- TRAIN ---
    epoch_loss = 0
    loop = tqdm(train_loader, desc="Training  ")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"→ Train Loss: {avg_train_loss:.4f}")

    # --- VALIDATION ---
    val_preds, val_labels, avg_val_loss = evaluate(val_loader, "Validating")
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1  = f1_score(val_labels, val_preds, average='binary')
    val_losses.append(avg_val_loss)
    print(f"→ Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Lưu model tốt nhất theo Val F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        os.makedirs("/app/model_best", exist_ok=True)
        model.save_pretrained("/app/model_best")
        tokenizer.save_pretrained("/app/model_best")
        print(f"→ Model tốt nhất lưu tại epoch {epoch+1} (F1: {val_f1:.4f})")

print(f"\n{'='*20} HUẤN LUYỆN HOÀN TẤT {'='*20}")
print(f"Train Loss: {[f'{x:.4f}' for x in train_losses]}")
print(f"Val Loss:   {[f'{x:.4f}' for x in val_losses]}")
print(f"Best Val F1: {best_val_f1:.4f}")
print(f"{'='*50}\n")

# ================================
# 10. TEST TRÊN TẬP TEST (CUỐI CÙNG)
# ================================
print(f"{'='*20} ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST {'='*20}")
test_preds, test_labels, _ = evaluate(test_loader, "Testing   ")
test_acc = accuracy_score(test_labels, test_preds)
test_f1  = f1_score(test_labels, test_preds, average='binary')
test_prec = precision_score(test_labels, test_preds, average='binary')
test_rec = recall_score(test_labels, test_preds, average='binary')

print(f"\n{'='*20} KẾT QUẢ TEST {'='*20}")
print(f"Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Precision : {test_prec:.4f}")
print(f"Recall    : {test_rec:.4f}")
print(f"F1-Score  : {test_f1:.4f}")
print("\nClassification Report (Test):")
print(classification_report(test_labels, test_preds, target_names=['Not Stress', 'Stress']))
print(f"{'='*50}\n")

# ================================
# 11. LƯU MODEL CUỐI CÙNG
# ================================
os.makedirs("/app/model_final", exist_ok=True)
model.save_pretrained("/app/model_final")
tokenizer.save_pretrained("/app/model_final")
print("Model cuối cùng lưu tại: /app/model_final")
print("Model tốt nhất (theo val) lưu tại: /app/model_best")
print("→ Cả 2 sẽ xuất hiện ở: E:\\HCMUTE\\cifar100\\stress_roberta\\")
print(f"{'='*50}")