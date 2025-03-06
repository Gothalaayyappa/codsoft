import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class ImageCaptioningDataset(Dataset):
    def __init__(self, image_paths, captions, vocab, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        tokens = word_tokenize(self.captions[idx].lower())
        caption = [self.vocab[token] for token in tokens if token in self.vocab]
        return image, torch.tensor(caption)

def build_vocab(captions):
    words = set(word for caption in captions for word in word_tokenize(caption.lower()))
    vocab = {word: idx + 1 for idx, word in enumerate(words)}
    vocab["<PAD>"] = 0
    itos = {idx: word for word, idx in vocab.items()}  # Reverse mapping
    return vocab, itos

class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CaptioningModel, self).__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.fc = nn.Linear(2048, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images).squeeze(-1).squeeze(-1)
        features = self.fc(features).unsqueeze(1)
        embeddings = self.embedding(captions)
        lstm_input = torch.cat((features, embeddings), dim=1)
        outputs, _ = self.lstm(lstm_input)
        outputs = self.fc_out(outputs)
        return outputs

def generate_caption(model, image, vocab, max_length=20):
    model.eval()
    image = image.unsqueeze(0)
    features = model.encoder(image).squeeze(-1).squeeze(-1)
    features = model.fc(features).unsqueeze(1)
    caption = []
    hidden = None
    input_word = torch.tensor([vocab['<PAD>']]).unsqueeze(0)
    for _ in range(max_length):
        embedding = model.embedding(input_word)
        lstm_input = torch.cat((features, embedding), dim=1)
        output, hidden = model.lstm(lstm_input, hidden)
        output = model.fc_out(output.squeeze(1))
        predicted = output.squeeze(0).argmax().item()  # Fixed here
        caption.append(predicted)
        if predicted == vocab['<PAD>']:
            break
        input_word = torch.tensor([predicted]).unsqueeze(0)
    return caption

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
vocab, itos = build_vocab(["A dog running", "A cat sleeping"])
dataset = ImageCaptioningDataset(
    [r"C:\Users\dell\Desktop\image1.jpeg", r"C:\Users\dell\Desktop\image2.jpeg"],
    ["A dog running", "A cat sleeping"],
    vocab,
    transform=transform
)
dataset.itos = itos  # Add index-to-word mapping

# DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)

# Initialize model
model = CaptioningModel(embed_size=256, hidden_size=512, vocab_size=len(dataset.vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for images, captions in dataloader:
        outputs = model(images, captions)  # Forward pass
        loss = criterion(
            outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1)),
            captions[:, 1:].contiguous().view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test with an image
test_image = transform(Image.open(r"C:\Users\dell\Desktop\image2.jpeg").convert("RGB"))
predicted_caption = generate_caption(model, test_image, dataset.vocab)
predicted_caption_words = [dataset.itos[idx] for idx in predicted_caption]  # Convert indices to words
print("Generated Caption:", " ".join(predicted_caption_words))
