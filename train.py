import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:  # Specify UTF-8 encoding
        return json.load(f)

# Load data from multiple JSON files
intents_data = load_json('intents.json')
products_data = load_json('products.json')
store_locator_data = load_json('storelocator.json')
promotion_data = load_json('promotions.json')
track_status_data = load_json('TrackStatus.json')
recommendation_data = load_json('recommendations.json')


all_words = []
tags = []
xy = []

# Process intents
for intent in intents_data['intents']:
    tag = intent['tag']
    if tag not in tags:
        tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Ensure 'product_inquiry' is in the tags list
if 'product_inquiry' not in tags:
    tags.append('product_inquiry')

# Process product information
products = {product['id']: product for product in products_data['products']}
for product in products_data['products']:
    xy.append((tokenize(product['name'].lower()), 'product_inquiry'))
    xy.append((tokenize(product['id'].lower()), 'product_inquiry'))
    for key, value in product.items():
        if isinstance(value, str):
            w = tokenize(value)
            all_words.extend(w)

# Ensure 'store_inquiry' is in the tags list
if 'store_inquiry' not in tags:
    tags.append('store_inquiry')

# Process store locator information
for store in store_locator_data['storelocator']:
    xy.append((tokenize(store['Location'].lower()), 'store_inquiry'))
    xy.append((tokenize(store['address'].lower()), 'store_inquiry'))
    xy.append((tokenize(store['phone'].lower()), 'store_inquiry'))
    xy.append((tokenize(store['hours'].lower()), 'store_inquiry'))
    xy.append((tokenize(store['website'].lower()), 'store_inquiry'))
    xy.append((tokenize(store['postal code'].lower()), 'store_inquiry'))
    for key, value in store.items():
        if isinstance(value, str):
            w = tokenize(value)
            all_words.extend(w)

# Ensure 'promotion_inquiry' is in the tags list
if 'promotion_inquiry' not in tags:
    tags.append('promotion_inquiry')

# Process promotion information
for promo in promotion_data['promotions']:
    promo_name = promo['name']
    promo_description = promo['description']
    promo_terms = promo.get('terms_conditions', '')  # Handle if 'terms_conditions' might be missing
    promo_start_date = promo.get('start_date', '')
    promo_end_date = promo.get('end_date', '')

    # Add promotion name and description to the training data
    xy.append((tokenize(promo_name.lower()), 'promotion_inquiry'))
    xy.append((tokenize(promo_description.lower()), 'promotion_inquiry'))
    xy.append((tokenize(promo_terms.lower()), 'promotion_inquiry'))
    xy.append((tokenize(promo_start_date.lower()), 'promotion_inquiry'))
    xy.append((tokenize(promo_end_date.lower()), 'promotion_inquiry'))

    # Add applicable products to the training data
    applicable_products = promo.get('applicable_products', {})
    
    # Check if applicable_products is a dictionary
    if isinstance(applicable_products, dict):
        for category, promotion_products in applicable_products.items():
            if isinstance(promotion_products, list):
                for promotion_product in promotion_products:
                    # Ensure the product has the required fields
                    if isinstance(promotion_product, dict):
                        # Adding product name and details
                        xy.append((tokenize(promotion_product.get('product_name', '').lower()), 'promotion_inquiry'))
                        xy.append((tokenize(promotion_product.get('original_price', '').lower()), 'promotion_inquiry'))
                        xy.append((tokenize(promotion_product.get('discount_price', '').lower()), 'promotion_inquiry'))
                        xy.append((tokenize(promotion_product.get('details', '').lower()), 'promotion_inquiry'))

                        # If you want to include product ID in the training data
                        xy.append((tokenize(promotion_product.get('product_id', '').lower()), 'promotion_inquiry'))
            else:
                print(f"Unexpected format for products under category '{category}': {promotion_products}")
    else:
        print(f"Unexpected format for applicable_products: {applicable_products}")
            
    
# Ensure 'track_status' is in the tags list
if 'track_status' not in tags:
    tags.append('track_status')

# Process track status information
for order in track_status_data['trackorder']:
    order_tokens = []
    for key, value in order.items():
        tokens = tokenize(str(value).lower())
        order_tokens.extend(tokens)
        all_words.extend(tokens)
    
    xy.append((order_tokens, 'track_status'))


# Initialize data dictionary
data = {}
# Ensure 'recommended_product_inquiry' is in the tags list
if 'recommended_product_inquiry' not in tags:
    tags.append('recommended_product_inquiry')

# Process recommendation data
if 'categories' in recommendation_data:
    for category, subcategories in recommendation_data['categories'].items():
        for subcategory, rec_products in subcategories.items():
            for rec_product_name, rec_product_details in rec_products.items():
                try:
                    # Use the rec_product_name as the product name
                    xy.append((tokenize(rec_product_name.lower()), 'recommended_product_inquiry'))
                    
                    # Process all details of the product
                    for key, value in rec_product_details.items():
                        if isinstance(value, str):
                            xy.append((tokenize(value.lower()), 'recommended_product_inquiry'))
                            w = tokenize(value)
                            all_words.extend(w)
                except Exception as e:
                    print(f"Error processing recommended product: {rec_product_name}. Error: {str(e)}")
else:
    print("Warning: 'categories' key not found in recommendation_data. Skipping recommendation processing.")

# Add recommendation data to the saved data
data["recommendations"] = recommendation_data


# Remove duplicates from all_words
all_words = list(set(all_words))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"Tags: {tags}")  # Debug print

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
    "products": products,
    "intents": intents_data['intents'],
    "store_locator": store_locator_data['storelocator'],
    "promotions": promotion_data['promotions'],
    "track_status": track_status_data['trackorder'],
    "recommendations": recommendation_data 
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')
