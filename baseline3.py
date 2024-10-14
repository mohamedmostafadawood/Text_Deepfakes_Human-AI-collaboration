# Install required libraries (uncomment to install)
# !pip install -q datasets evaluate transformers sentencepiece matplotlib seaborn scikit-learn


"""             precision    recall  f1-score   support

       Human       0.00      0.00      0.00      8000
          AI       0.50      1.00      0.67      8000

    accuracy                           0.50     16000
   macro avg       0.25      0.50      0.33     16000
weighted avg       0.25      0.50      0.33     16000    """

Human examples: 177, AI examples: 321
Classification Report:
              precision    recall  f1-score   support

       Human       1.00      0.00      0.00       177
          AI       0.64      1.00      0.78       321

    accuracy                           0.64       498
   macro avg       0.82      0.50      0.39       498
weighted avg       0.77      0.64      0.51       498

Accuracy on new dataset: 0.6446

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
from transformers import LlamaForSequenceClassification, AutoTokenizer, get_scheduler, set_seed
from huggingface_hub import login

# Login to Hugging Face
login("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")

# Set seed for reproducibility
set_seed(0)

# Hyperparameters
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
ACCUMULATION_STEPS = 4
LR = 1e-4
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 1
SEED = 42
MODEL_MAX_LENGTH = 64

# Tokenizer and Model Setup
def setup_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = MODEL_MAX_LENGTH
    tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForSequenceClassification.from_pretrained(
        model_name, num_labels=2  # Binary classification (human vs AI)
    ).to(DEVICE)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

tokenizer, model = setup_tokenizer_and_model(MODEL_NAME)

# Load and Preprocess Dataset
def load_and_preprocess_dataset():
    # Load Sentiment140 dataset
    human_dataset = load_dataset("sentiment140", split="train[:5%]")  # 5% of dataset for demonstration

    # Filter out empty or invalid text
    def filter_empty_text(examples):
        return len(examples['text'].strip()) > 0
    
    human_dataset = human_dataset.filter(filter_empty_text)

    # Create AI-generated dataset
    ai_generated_dataset = human_dataset.map(lambda x: {"text": "This is an AI-generated text based on: " + x['text']})

    # Assign labels
    human_dataset = human_dataset.map(lambda x: {"label": 0})
    ai_generated_dataset = ai_generated_dataset.map(lambda x: {"label": 1})

    # Combine datasets and shuffle
    balanced_dataset = concatenate_datasets([human_dataset, ai_generated_dataset]).shuffle(seed=SEED)

    # Tokenize dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding='max_length', truncation=True, return_token_type_ids=False)

    tokenized_datasets = balanced_dataset.map(preprocess_function, batched=True)
    
    # Keep only necessary columns
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "date", "query", "user"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets

tokenized_datasets = load_and_preprocess_dataset()

# Stratified Split for Train/Validation
def stratified_split(dataset, test_size=0.1, seed=42):
    labels = [example['labels'].item() for example in dataset]
    train_idx, val_idx = train_test_split(range(len(labels)), test_size=test_size, random_state=seed, stratify=labels)
    train_dataset = dataset.select(train_idx)
    valid_dataset = dataset.select(val_idx)
    return train_dataset, valid_dataset

train_dataset, valid_dataset = stratified_split(tokenized_datasets)

# DataLoader setup
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# Reward Function
def calculate_reward(predictions, ground_truth):
    # Define a reward that is based purely on the correctness of prediction
    correct_predictions = (predictions == ground_truth).float()
    return correct_predictions  # Simple reward based on correct predictions

# Training Loop with DPO
def train_with_dpo(model, train_dataloader, valid_dataloader):
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * NUM_EPOCHS
    )
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_reward = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Model outputs predictions (logits)
            outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
            logits = outputs.logits
            predictions = logits.argmax(dim=-1)

            # Calculate reward based on predictions
            rewards = calculate_reward(predictions, batch['labels'])

            # Convert logits to log probabilities (log_softmax)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Create a custom loss using log probabilities scaled by the reward
            # Gather the log probability for the predicted class
            log_probs_for_predicted = log_probs[range(log_probs.shape[0]), predictions]
            
            # Negate log probs and multiply by rewards (we negate to maximize reward)
            custom_loss = -(log_probs_for_predicted * rewards).mean()

            # Backpropagation based on the custom loss
            custom_loss.backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_train_reward += rewards.mean().item()

        # Average training reward for the epoch
        avg_train_reward = total_train_reward / len(train_dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Training Reward: {avg_train_reward:.4f}")

        # Validation loop
        avg_val_reward = evaluate_on_validation(model, valid_dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Reward: {avg_val_reward:.4f}")

# Evaluation on Validation Set
def evaluate_on_validation(model, valid_dataloader):
    model.eval()
    total_val_reward = 0

    for batch in valid_dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
        
        predictions = outputs.logits.argmax(dim=-1)
        rewards = calculate_reward(predictions, batch['labels'])
        total_val_reward += rewards.mean().item()

    avg_val_reward = total_val_reward / len(valid_dataloader)
    return avg_val_reward

# Training the model using DPO
train_with_dpo(model, train_dataloader, valid_dataloader)

# Save the trained model
torch.save(model.state_dict(), "llama_model_dpo.pt")
print("Model saved locally as 'llama_model_dpo.pt'")

# Confusion Matrix and Classification Report
def plot_confusion_matrix_and_report(model, valid_dataloader):
    model.eval()
    all_labels = []
    all_predictions = []

    for batch in valid_dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
        predictions = outputs.logits.argmax(dim=-1)
        all_labels.extend(batch['labels'].cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Human', 'AI']))

# Plot confusion matrix and report
plot_confusion_matrix_and_report(model, valid_dataloader)
