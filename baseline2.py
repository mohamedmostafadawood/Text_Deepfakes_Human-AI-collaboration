# Install required libraries
# !pip install -q datasets evaluate transformers sentencepiece matplotlib seaborn scikit-learn
"""100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9000/9000 [58:32<00:00,  2.56it/s]
Epoch 1/2, Training Loss: 0.0049
Epoch 1/2, Validation Accuracy: 1.0000
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9000/9000 [58:32<00:00,  2.56it/s]
Epoch 2/2, Training Loss: 0.0000
Epoch 2/2, Validation Accuracy: 1.0000
Classification Report:
              precision    recall  f1-score   support

       Human       1.00      1.00      1.00     16000
          AI       1.00      1.00      1.00     16000

    accuracy                           1.00     32000
   macro avg       1.00      1.00      1.00     32000
weighted avg       1.00      1.00      1.00     32000"""


""" Human examples: 177, AI examples: 321
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 498/498 [00:00<00:00, 20821.42 examples/s]
Classification Report:
              precision    recall  f1-score   support

       Human       0.36      1.00      0.52       177
          AI       1.00      0.00      0.00       321

    accuracy                           0.36       498
   macro avg       0.68      0.50      0.26       498
weighted avg       0.77      0.36      0.19       498

Accuracy on new dataset: 0.3554"""

import torch
import transformers
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, LlamaForSequenceClassification, get_scheduler, set_seed
from huggingface_hub import login
from transformers import AutoTokenizer

login("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")

set_seed(0)

# Hyperparameters for training
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16  # Increase batch size slightly
ACCUMULATION_STEPS = 4  # Reduce gradient accumulation steps
LR = 1e-4  # Reduced learning rate to prevent exploding gradients
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 2
SEED = 42
MODEL_MAX_LENGTH = 64

# Prepare tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.padding_side = 'right'
tokenizer.model_max_length = MODEL_MAX_LENGTH
tokenizer.pad_token = tokenizer.eos_token

# Load the model locally for sequence classification
model = LlamaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2  # Set num_labels to 2 for binary classification
).to(DEVICE)

model.config.pad_token_id = tokenizer.pad_token_id

# Load the Sentiment140 dataset from Hugging Face (for human-written text)
human_dataset = load_dataset("sentiment140", split="train[:5%]")  # Using 5% for demonstration

# Filter out empty or invalid text from the dataset
def filter_empty_text(examples):
    return len(examples['text'].strip()) > 0

human_dataset = human_dataset.filter(filter_empty_text)

# Simulate AI-generated text by copying the human-written dataset
ai_generated_dataset = human_dataset.map(lambda x: {"text": "This is an AI-generated text based on: " + x['text']})

# Assign labels (0 for human, 1 for AI-generated)
human_dataset = human_dataset.map(lambda x: {"label": 0})
ai_generated_dataset = ai_generated_dataset.map(lambda x: {"label": 1})

# Combine the two datasets using `concatenate_datasets`
balanced_dataset = concatenate_datasets([human_dataset, ai_generated_dataset]).shuffle(seed=SEED)

# Preprocess dataset (tokenization)
def preprocess_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, return_token_type_ids=False)

tokenized_datasets = balanced_dataset.map(preprocess_function, batched=True)

# Keep only 'input_ids' and 'labels' for the model
tokenized_datasets = tokenized_datasets.remove_columns(["text", "date", "query", "user"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Stratified split to ensure both training and validation datasets are balanced
def stratified_split(dataset, test_size=0.1, seed=42):
    labels = [example['labels'].item() for example in dataset]
    train_idx, val_idx = train_test_split(range(len(labels)), test_size=test_size, random_state=seed, stratify=labels)
    
    train_dataset = dataset.select(train_idx)
    valid_dataset = dataset.select(val_idx)
    
    return train_dataset, valid_dataset

# Perform stratified split
train_dataset, valid_dataset = stratified_split(tokenized_datasets)

# DataLoader for training and validation sets
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# Load accuracy metric
metric = evaluate.load('accuracy')

# Prepare optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * NUM_EPOCHS
)

# Mixed precision training setup (DISABLED TEMPORARILY)
scaler = torch.amp.GradScaler()

# Tracking variables for plotting
train_losses = []
val_accuracies = []
all_labels = []
all_predictions = []

# Training loop without mixed precision for debugging
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
        loss = outputs.loss / ACCUMULATION_STEPS
        loss.backward()

        if (step + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.detach().item()

    # Log training loss
    avg_train_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {avg_train_loss:.4f}")

    # Evaluate after each epoch
    model.eval()
    correct = 0
    total = 0
    for batch in valid_dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)
        all_labels.extend(batch['labels'].cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Accuracy: {val_accuracy:.4f}")

# Plotting training loss and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker='o', label='Training Loss')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, marker='o', label='Validation Accuracy', color='green')
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Confusion matrix to analyze bias
cm = confusion_matrix(all_labels, all_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=['Human', 'AI']))

# Save the model
torch.save(model.state_dict(), "llama_model.pt")
print("Model saved locally as 'llama_model.pt'")
