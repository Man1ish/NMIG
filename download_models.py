import torch
import torchvision.models as models
from transformers import BertModel
import os
from transformers import BertModel, BertTokenizer,AutoModelForCausalLM, AutoTokenizer

# Define the models and their filenames
model_list = [
    (models.alexnet, "alexnet.pth"),
    (models.resnet50, "resnet50.pth"),
    (models.efficientnet_b7, "efficientnet_b7.pth"),
    (models.inception_v3, "inception_v3.pth"),
    (models.googlenet, "googlenet.pth"),
]

save_dir = "python_runtime/core/python39Action/models/"

# Download and save torchvision models
for model_func, filename in model_list:
    print(f"Downloading {filename}...")
    model = model_func(pretrained=True)  # Download pretrained weights
    torch.save(model.state_dict(), save_dir + filename)
    print(f"{filename} has been downloaded and saved.")

# === BERT BASE (UNCASED) ===
print("Downloading BERT (bert-base-uncased)...")

bert_path = os.path.join(save_dir, "bert-base-uncased")
os.makedirs(bert_path, exist_ok=True)

bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert_model.save_pretrained(bert_path)
bert_tokenizer.save_pretrained(bert_path)

print("BERT model and tokenizer saved to:", bert_path)


# === DISTILGPT2 ===
print("Downloading DistilGPT2...")

gpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

gpt2_path = os.path.join(save_dir, "distilgpt2_local")
os.makedirs(gpt2_path, exist_ok=True)

# Save model and tokenizer
gpt2_model.save_pretrained(gpt2_path)
gpt2_tokenizer.save_pretrained(gpt2_path)

# Save model state_dict as .pth
gpt2_pth_path = os.path.join(save_dir, "distilgpt2.pth")
torch.save(gpt2_model.state_dict(), gpt2_pth_path)

print("DistilGPT2 model and tokenizer saved to:", gpt2_path)
print("distilgpt2.pth saved to:", gpt2_pth_path)

print("All models have been successfully downloaded and saved.")
