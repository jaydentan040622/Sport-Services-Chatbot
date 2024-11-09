import random
import torch
import json
import re
import time
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from difflib import SequenceMatcher

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
intents = data["intents"]
products = data["products"]
store_locator_data = data.get("store_locator", [])
promotions_data = data.get("promotions", [])

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
current_product = None
current_promotion_context = None
current_order_context = None
current_context = None
current_recommendation_context = None

# Define variants for different query types
color_variants = ["color", "colour", "shade", "colors", "colours"]
price_variants = ["price", "cost", "worth", "value"]
availability_variants = ["availability", "available", "in stock", "stock"]
description_variants = ["description", "details", "info", "information", "about", "tell me more", "overview"]
store_query_variants = ["store", "location", "address", "shop", "place"]
promotion_variants = ["promotion", "sale", "discount", "offer", "deal", "promotions"]
trackstatus_variants = ["track", "status", "order", "delivery", "shipping", "parcel", "tracking"]
validity_variants = ["validity", "date", "period", "duration", "how long", "when"]
product_variants = ["products", "items", "what's included", "what is included", "apply to"]
terms_variants = ["terms", "conditions", "rules", "fine print", "terms and condition", "terms and conditions"]
recommendation_variants = ["recommend", "suggestion", "suggest", "what do you recommend", "show me"]
category_variants = ["category", "type", "kind", "sort"]
detail_variants = ["detail", "more info", "tell me more", "specifications"]
promotion_product_variants = ["product", "item", "what's included", "what is included", "apply to"]


# Load store locator data
with open('StoreLocator.json', 'r') as file:
    store_locator_data = json.load(file)['storelocator']

# Load recommendation data
with open('recommendations.json', 'r') as file:
    recommendation_data = json.load(file)['recommendations']

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def is_similar(word, options, threshold=0.7):
    return any(similar(word, option) > threshold for option in options)

def get_random_response(responses):
    return random.choice(responses)

#-------------------------------------------------------------------------------------------
# Main response function
# Main response function
def get_response(msg):
    global current_product, current_promotion_context, current_context, current_order_context
    global products, store_locator_data  # Declare products and other variables as global

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    msg_lower = ' '.join(sentence).lower()
    

    if prob.item() > 0.8:  # Lower probability threshold
        # Product inquiry handling
        if tag == "product_inquiry" or current_product:
            product_response = handle_product_inquiry(sentence, msg_lower)
            if product_response:
                return product_response

        # Handle promotion queries
        if tag == "promotion_inquiry" or any(is_similar(word, promotion_variants) for word in sentence):
            promotion_response = handle_promotion_query(sentence, msg_lower, prob, tag)
            if promotion_response:
                return promotion_response

    # Check for specific store names first        
    for store in store_locator_data:
        if store['Location'].lower() in msg_lower:
            return get_store_details(store['Location'])
           

    # Check for recommendation query
    if any(variant in msg_lower for variant in recommendation_variants):
        return handle_recommendation(msg_lower)

    # If we're in the middle of a recommendation flow, continue handling it
    if current_recommendation_context:
        return handle_recommendation(msg_lower)

    # Store locator and general intents
    location_keywords = {
        "kl": "Kuala Lumpur",
        "kuala lumpur": "Kuala Lumpur",
        "penang": "Penang",
        "selangor": "Selangor",
        "melaka": "Melaka"
    }

    msg_lower = msg.lower()

    # Store locator handling
    if prob.item() > 0.5:
        if any(keyword in msg_lower for keyword in ["store", "shop", "location", "adidas store"]):
            # Check for specific store query
            specific_store = next((store for store in store_locator_data if store['Location'].lower() in msg_lower), None)
            if specific_store:
                return get_store_details(specific_store['Location'])
            
            for keyword, location in location_keywords.items():
                if keyword in msg_lower:
                    return list_stores_by_location(location)
            
            if any(keyword in msg_lower for keyword in ["nearest", "closest", "nearby", "around me", "near me", "close by"]):
                zipcode = next((word for word in msg_lower.split() if word.isdigit() and len(word) == 5), None)
                if zipcode:
                    return find_nearest_store(zipcode)
                else:
                    return get_random_response([
                            [
                                "🏙️ Ready to find your nearest store? I just need a little help from you!",
                                "🔢 Could you please share your 5-digit zipcode? For example, you could say:",
                                "'Find the nearest store to 12345' 😊"
                            ],
                            [
                                "🗺️ Let's locate the perfect store for you! I just need one tiny detail.",
                                "📮 What's your 5-digit zipcode? You can simply say something like:",
                                "'What's the closest store to 67890?' 👍"
                            ],
                            [
                                "🏬 Excited to help you find your nearest store! I just need a small clue.",
                                "🧭 Mind sharing your 5-digit zipcode? Try saying something like:",
                                "'Is there a store near 54321?' 😄"
                            ]
                        ])
                
            return get_random_response([
                [
                    "🌆 Are you on the hunt for a store in a specific area? I'm all ears!",
                    "🌍 You can ask about stores in any city, or if you prefer, give me a zipcode for the nearest location.",
                    "What works best for you? 😊"
                ],
                [
                    "🏙️ Looking for a store somewhere special? I'm here to help!",
                    "🔍 Feel free to ask about stores in a particular city, or share a zipcode to find the closest one.",
                    "Which would you like to try? 👀"
                ],
                [
                    "🗺️ Ready to locate a store? Let's make it happen!",
                    "🌎 You can ask about stores in a specific city, or give me a zipcode to find the nearest spot.",
                    "What's your preference? 🤔"
                ]
            ])
        
    # Handle track status queries
    if any(word in msg_lower for word in trackstatus_variants):
        current_context = "awaiting_order_id_or_email"
        return [
            "Certainly! I can help you track your order.🚚",
            "Could you please provide your order ID (starting with 'ORD') or the email address associated with your order?🤔"
        ]

    # Handle response to order tracking prompt
    if current_context in ["awaiting_order_id", "awaiting_order_id_or_email"]:
        order_id = extract_order_id(msg)
        if order_id:
            current_context = None
            track_status_result = get_track_status(order_id)
            current_order_context = {
                'order_data': track_status_result['order_data'],
                'next_question': track_status_result['next_question']
            }
            return track_status_result['response']
        elif is_valid_email(msg):
            orders = get_orders_by_email(msg)
            if orders:
                response = [f"📦 Great news! I found {len(orders)} order(s) associated with the email {msg}:"]
                for i, order in enumerate(orders, 1):
                    response.append(f"\n🛍️ Order {i}:\n{format_order_summary(order)}")
                response.append("\n🔍 Which order would you like more details on? Please provide the order ID.")
                current_context = "awaiting_order_id"
                return response
            else:
                current_context = None
                return ["🤔 Hmm, I couldn't find any orders associated with that email address. Could you please double-check the email and try again?"]

    # Handle follow-up questions for order tracking
    if current_order_context and current_order_context['order_data']:
        followup_response, next_question = handle_order_followup(msg, current_order_context['order_data'], current_order_context['next_question'])
        current_order_context['next_question'] = next_question
        if next_question is None:
            current_order_context = None
        return followup_response
    
    # Intent handling
    if prob.item() > 0.75:
        for intent in intents:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "🤷‍♂️ I'm sorry, I'm not sure how to help with that. Could you please provide more information?"



#----------------------------------------------------------------------------------------------------------------------------------------------
# Product info functions
def get_product_info(product_id, attribute):
    product = products.get(product_id)
    if product:
        return product.get(attribute, "I don't have that information for this product.")
    return "I couldn't find information about that product."

def handle_product_inquiry(sentence, msg_lower):
    global current_product  # Ensure current_product is accessible globally
    
    # Split the sentence into words for easier matching
    words = msg_lower.split()

    # If no current product is set, attempt to find one
    if not current_product:
        best_match = None

        # Iterate over the products to find the best match
        for product_id, product_info in products.items():
            product_text = (product_info['name'] + ' ' + product_info['category']).lower()
            
            # Calculate similarity between words in query and product info
            similarities = [similar(word, prod_word) for word in words for prod_word in product_text.split()]
            if similarities:
                max_similarity = max(similarities)
                if max_similarity > 0.8:  # Set a high similarity threshold
                    best_match = (product_id, max_similarity)
                    break  # Exit once a product match is found

        if best_match:
            # Set the current product to the best match
            current_product = best_match[0]
            product = products[current_product]
            product_string = f"{product['name']} (ID: {current_product})"

            # Respond with product information
            return get_random_response([
                [
                    f"🎉 Great! I found a product that might interest you: {product_string}.",
                    "Would you like to know more about it? You can ask about color, price, availability, or get a general overview. 😊"
                ],
                [
                    f"🌟 Excellent! Here's a product that matches your query: {product_string}.",
                    "Feel free to ask about color, price, stock status, or request a general description. 😄"
                ],
                [
                    f"👀 I've got a great option for you! Check out {product_string}.",
                    "What would you like to explore? You can ask about color, price, availability, or get an overview. 🤓"
                ]
            ])
        else:
            # No product match found
            return get_random_response([
                [
                    "I'm sorry, I couldn't find an exact match for that product.",
                    "Could you please try describing it differently or provide more details? 😊"
                ],
                [
                    "Hmm, I'm not finding a product that matches your description.",
                    "Could you rephrase your request or give me more information? 🤔"
                ],
                [
                    "I apologize, but I'm having trouble locating a product based on that description.",
                    "Could you try using different keywords or be more specific? 🧐"
                ]
            ])

    # If a product has already been selected, handle specific inquiries
    if current_product:
        product = products[current_product]
        responses = {
            'color': [
                [
                    f"🌈 The {product['name']} comes in a stunning {product['color']}.",
                    "Quite eye-catching, isn't it? 😍 Is there anything else you'd like to know?"
                ],
                [
                    f"🎨 The {product['name']} is available in a gorgeous {product['color']}.",
                    "What else can I tell you about it? Interested in the price or availability?"
                ]
            ],
            'price': [
                [
                    f"💰 The {product['name']} is priced at {product['selling_price']}.",
                    "Would you like to know more about its features or perhaps explore other products?"
                ],
                [
                    f"🏷️ The {product['name']} is going for {product['selling_price']}.",
                    "Anything else you'd like to know?"
                ]
            ],
            'availability': [
                [
                    f"📦 The {product['name']} is {product['availability']}.",
                    "Feel free to ask about colors, pricing, or other details."
                ]
            ],
            'description': [
                [
                    f"📝 Here's the scoop on the {product['name']}: {product['description']}",
                    "Is there anything specific you'd like to know?"
                ]
            ]
        }

        # Check if the inquiry matches one of the product attributes (color, price, etc.)
        for attribute, variants in [('color', color_variants), ('price', price_variants),
                                    ('availability', availability_variants), ('description', description_variants)]:
            if any(is_similar(word, variants) for word in sentence):
                return get_random_response(responses[attribute])

    return None

# -----------------------------------------------------------------------------------------------------
#promotion 
def get_promotion_names():
    return [promo['name'] for promo in promotions_data]

def get_promotion_info(promotion_name):
    if isinstance(promotion_name, dict):
        promotion_name = promotion_name.get('name', '')
        
    for promo in promotions_data:
        if similar(promo['name'].lower(), promotion_name.lower()) > 0.8:
            return promo
    return None

def get_promotion_detail(promotion_name, detail):
    promo = get_promotion_info(promotion_name)
    if promo:
        if detail == 'validity':
            return f"from {promo['start_date']} to {promo['end_date']}"
        elif detail in promo:
            return promo[detail]
    return None

def get_applicable_products(promotion_name):
    """Returns the applicable products for a given promotion."""
    promo = get_promotion_info(promotion_name)
    if promo and 'applicable_products' in promo:
        return promo['applicable_products']
    return {}


def get_product_details(promotion_name, category, product_id):
    promo = get_promotion_info(promotion_name)
    if promo:
        for prod in promo['applicable_products'].get(category, []):
            if prod['product_id'] == product_id:
                return prod
    return {}

def get_promotion_terms(promotion_name):
    promo = get_promotion_info(promotion_name)
    if promo:
        return promo['terms_conditions']
    return None

def handle_promotion_query(sentence, msg_lower, prob, tag):
    global current_promotion_context
    # Ensure current_promotion_context is a dictionary at the start
    current_promotion_context = current_promotion_context if isinstance(current_promotion_context, dict) else {'name': current_promotion_context}
    
    if current_promotion_context and isinstance(current_promotion_context, dict):
        applicable_products = current_promotion_context.get('applicable_products', {})
    else:
        applicable_products = {}
    
    if prob.item() > 0.8:
        # Check if the sentence is about current promotions
        if tag == "promotion_inquiry" or any(is_similar(word, promotion_variants) for word in sentence):
            if any(word in msg_lower for word in ["recent", "current", "ongoing", "latest", "new", "active"]):
                promo_names = get_promotion_names()
                return [
                    "🎉 We've got some great promotions running right now.",
                    f"🔥 Here's what's available: {', '.join(promo_names)}.",
                    "👀 Any of these interest you? I can provide more details!"
                ]

            # Check for a specific promotion
            for promo in promotions_data:
                promo_words = promo['name'].lower().split()
                if any(similar(word, promo_word) > 0.8 for word in sentence for promo_word in promo_words):
                    current_promotion_context = promo['name']
                    return [
                        f"🎉 You're interested in the {promo['name']} promotion! Here's the scoop:",
                        f"{promo['description']}",
                        "You can ask about:",
                        "📅 The validity period",
                        "🛍️ Applicable products",
                        "📋 Specific terms and conditions",
                        "Just let me know what interests you! 😊"
                    ]

            # If a promotion context is set, handle specific queries
            if current_promotion_context:
                if any(is_similar(word, description_variants) for word in sentence):
                    promo_info = get_promotion_info(current_promotion_context)
                    return get_random_response([
                        f"📢 Here's the details on the {current_promotion_context}: {promo_info['description']}. Anything else you're curious about? 😊"
                    ])

                elif any(is_similar(word, validity_variants) for word in sentence):
                    promo_name = current_promotion_context['name'] if isinstance(current_promotion_context, dict) else current_promotion_context
                    validity = get_promotion_detail(promo_name, 'validity')
                    return get_random_response([
                        f"⏰ The {promo_name} is valid {validity}. Don't miss out!"
                    ])

                elif any(is_similar(word, terms_variants) for word in sentence):
                    promo_name = current_promotion_context['name'] if isinstance(current_promotion_context, dict) else current_promotion_context
                    terms = get_promotion_terms(promo_name)
                    if terms:
                        return get_random_response([
                            f"📋 The terms for {promo_name}: {terms}. Any questions?"
                        ])
                    else:
                        return get_random_response([
                            "😕 Oops! I can't find the terms for this promotion."
                        ])

                # Handle queries about what the promotion includes
                elif any(word in msg_lower for word in ["include", "what is include", "products", "apply to", "items", "what's included"]):
                    applicable_products = get_applicable_products(current_promotion_context['name'])
                    
                    if applicable_products:
                        # Store applicable products in the promotion context
                        current_promotion_context['applicable_products'] = applicable_products
                        
                        # Return applicable product categories
                        categories = ', '.join(applicable_products.keys())
                        return [
                            f"The {current_promotion_context['name']} applies to these categories: {categories}. Do you want to know more about products in a specific category?"
                        ]
                    else:
                        return [
                            "I'm not sure which products this promotion applies to right now. Can I help you with something else?"
                        ]

# Handle further queries about a specific product category in the promotion
    # Handle further queries about a specific category in the promotion
    applicable_products = current_promotion_context.get('applicable_products', {})
    if applicable_products:
        for category in applicable_products:
            if category.lower() in msg_lower:
                products = applicable_products[category]

                # List the product names
                product_names = [p.get('product_name', 'Unknown Product') for p in products]
                return [
                    f"Sam: In the {category} category, the following products are included: {', '.join(product_names)}. Which product would you like details on?"
                ]
                
    # Handle specific product query
    for category, products in applicable_products.items():
        for product in products:
            if product.get('product_name', '').lower() in msg_lower:
                # Provide product details
                product_name = product.get('product_name', 'Unknown Product')
                original_price = product.get('original_price', 'N/A')
                discount_price = product.get('discount_price', 'N/A')
                description = product.get('details', 'No description available')

                return [
                    f"The {product_name} is currently priced at {discount_price} (original price: {original_price}). {description}"
                ]
    
#------------------------------------------------------------------------------------------------
# Recommendation functions
def get_recommendation_categories():
    return list(recommendation_data['categories'].keys())

def get_recommendation_subcategories(category):
    return list(recommendation_data['categories'][category].keys())

def get_recommendation_products(category, subcategory):
    return list(recommendation_data['categories'][category][subcategory].keys())

def get_recommendation_product_details(category, subcategory, product_id):
    return recommendation_data['categories'][category][subcategory][product_id]

def handle_recommendation(msg):
    global current_recommendation_context

    # Check for exit keywords
    exit_keywords = ['exit', 'quit', 'stop', 'change topic', 'no thanks', 'not interested',"no"]
    if any(keyword in msg.lower() for keyword in exit_keywords):
        current_recommendation_context = None
        return ["Sure, let's change the topic. What else can I help you with? 😊"]

    if not current_recommendation_context:
        categories = get_recommendation_categories()
        current_recommendation_context = {"step": "category"}
        category_list = ", ".join(categories[:-1]) + f", and {categories[-1]}"
        return [
            "🛍️ I'd be thrilled to help you find the perfect product!",
            f"We have a fantastic selection in {category_list}.",
            "Which category catches your eye? 😊 "
        ]

    if current_recommendation_context["step"] == "category":
        category = next((cat for cat in get_recommendation_categories() if cat.lower() in msg.lower()), None)
        if category:
            current_recommendation_context["category"] = category
            subcategories = get_recommendation_subcategories(category)
            current_recommendation_context["step"] = "subcategory"
            subcategory_list = ", ".join(subcategories[:-1]) + f", and {subcategories[-1]}"
            return [
                f"🌟 Excellent choice! {category} is very popular right now.",
                f"Within {category}, we have {subcategory_list}.",
                "Which subcategory would you like to explore? 👀 "
            ]
        else:
            categories = get_recommendation_categories()
            category_list = ", ".join(categories[:-1]) + f", and {categories[-1]}"
            return [
                "😅 Oops! I didn't quite catch that. Let's try again.",
                f"Could you please choose from {category_list}?",
                "Which one interests you? 🤔"
            ]

    if current_recommendation_context["step"] == "subcategory":
        category = current_recommendation_context["category"]
        subcategory = next((sub for sub in get_recommendation_subcategories(category) if sub.lower() in msg.lower()), None)
        if subcategory:
            current_recommendation_context["subcategory"] = subcategory
            products = get_recommendation_products(category, subcategory)
            current_recommendation_context["step"] = "product"
            product_names = [recommendation_data['categories'][category][subcategory][p]['name'] for p in products[:3]]
            product_list = ", ".join(product_names[:-1]) + f", and {product_names[-1]}" if len(product_names) > 1 else product_names[0]
            return [
                f"🎉 Great! Here are some of our top picks in {subcategory}:",
                f"We have the {product_list}.",
                "Which product would you like more details on? 😊"
                
            ]
        else:
            subcategories = get_recommendation_subcategories(category)
            subcategory_list = ", ".join(subcategories[:-1]) + f", and {subcategories[-1]}"
            return [
                "🤔 I apologize, but I didn't recognize that subcategory.",
                f"Please choose from {subcategory_list}."
            ]

    if current_recommendation_context["step"] == "product":
        category = current_recommendation_context["category"]
        subcategory = current_recommendation_context["subcategory"]
        products = recommendation_data['categories'][category][subcategory]
        product = next((p for p, info in products.items() if info['name'].lower() in msg.lower()), None)
        if product:
            current_recommendation_context["product"] = product
            product_details = products[product]
            current_recommendation_context["step"] = "details"
            return [
                f"🌟 Excellent choice! Here's what you need to know about the {product_details['name']}:",
                f"📝 Description: {product_details['description']}",
                f"💰 Price: {product_details['selling_price']}",
                f"🎨 Colors: {product_details['color']}",
                f"📦 Availability: {product_details['availability']}",
                "Would you like to know more about the sizes, rating, or features? 😊"

            ]
        else:
            return [
                "🤨 I'm sorry, but I couldn't find that product.",
                "Could you please try again with the product name?"
            ]

    if current_recommendation_context["step"] == "details":
        product_details = get_recommendation_product_details(
            current_recommendation_context["category"],
            current_recommendation_context["subcategory"],
            current_recommendation_context["product"]   
        )
        if "size" in msg.lower():
            size_list = ", ".join(product_details['sizes'][:-1]) + f", and {product_details['sizes'][-1]}"
            return [
                f"📏 The {product_details['name']} is available in these sizes:",
                size_list,
                "Would you like to know about the rating or features? 😊"
            ]
        elif "rating" in msg.lower():
            return [
                f"⭐ The {product_details['name']} has received a rating of {product_details['rating']}/5 stars from our customers.",
                "Would you like to know about the sizes or features? 😊"
            ]
        elif "feature" in msg.lower():
            feature_list = ", ".join(product_details['features'][:-1]) + f", and {product_details['features'][-1]}"
            current_recommendation_context = None  # Reset context
            return [
                f"✨ Here are the standout features of the {product_details['name']}:",
                feature_list,
                "Is there anything else you'd like to know about our products? 😊",
            ]
        else:
            return [
                "😅 I'm sorry, I didn't catch that."
            ]

    current_recommendation_context = None
    return [
        "I apologize for the confusion. Let's start over.",
        "What kind of product information are you looking for? 😊",

    ]

# -----------------------------------------------------------------------------------------------------
# Store locator functions
def handle_store_locator(msg_lower):
    location_keywords = {
        "kl": "Kuala Lumpur",
        "kuala lumpur": "Kuala Lumpur",
        "penang": "Penang",
        "selangor": "Selangor",
        "melaka": "Melaka"
    }

    # Check for specific location
    for keyword, location in location_keywords.items():
        if keyword in msg_lower:
            return list_stores_by_location(location)

    # Handle requests for the nearest store
    if any(keyword in msg_lower for keyword in ["nearest", "closest", "nearby", "around me", "near me", "close by"]):
        zipcode = next((word for word in msg_lower.split() if word.isdigit() and len(word) == 5), None)
        if zipcode:
            return find_nearest_store(zipcode)
        else:
            return get_random_response([
                [
                    "🏙️ Ready to find your nearest store? I just need a little help from you!",
                    "🔢 Could you please share your 5-digit zipcode? For example, you could say:",
                    "'Find the nearest store to 12345' 😊"
                ],
                [
                    "🗺️ Let's locate the perfect store for you! I just need one tiny detail.",
                    "📮 What's your 5-digit zipcode? You can simply say something like:",
                    "'What's the closest store to 67890?' 👍"
                ],
                [
                    "🏬 Excited to help you find your nearest store! I just need a small clue.",
                    "🧭 Mind sharing your 5-digit zipcode? Try saying something like:",
                    "'Is there a store near 54321?' 😄"
                ]
            ])

    # If no specific location or zipcode is mentioned
    return get_random_response([
        [
            "🌆 Are you on the hunt for a store in a specific area? I'm all ears!",
            "🌍 You can ask about stores in any city, or if you prefer, give me a zipcode for the nearest location.",
            "What works best for you? 😊"
        ],
        [
            "🏙️ Looking for a store somewhere special? I'm here to help!",
            "🔍 Feel free to ask about stores in a particular city, or share a zipcode to find the closest one.",
            "Which would you like to try? 👀"
        ],
        [
            "🗺️ Ready to locate a store? Let's make it happen!",
            "🌎 You can ask about stores in a specific city, or give me a zipcode to find the nearest spot.",
            "What's your preference? 🤔"
        ]
    ])

def list_stores_by_location(location):
    stores = [f"- {store.get('Location')}"
              for store in store_locator_data
              if location.lower() in store.get('address', '').lower()]
    if stores:
        return get_random_response([
            [
                f"🏙️ Great news! I've found some stores in {location} for you:",
                "\n".join(f"🏬 {store}" for store in stores),
                "Which one would you like to know more about? Just ask! 😊"
            ],
            [
                f"🗺️ Look at all these options in {location}!",
                "\n".join(f"🏪 {store}" for store in stores),
                "Any of these catch your eye? I can give you more details on any of them! 👀"
            ],
            [
                f"🌟 You're in luck! Here are the stores I found in {location}:",
                "\n".join(f"🛍️ {store}" for store in stores),
                "Want to know more about a specific store? Just say the word! 🗣️"
            ]
        ])
    else:
        return get_random_response([
            [
                f"😕 Oops! It looks like {location} is playing hide and seek with our stores.",
                "🔍 I couldn't find any stores listed there at the moment.",
                "Want to try searching in a nearby area? I'm here to help! 🌟"
            ],
            [
                f"🤔 Hmm, I'm drawing a blank on stores in {location} right now.",
                "📍 It seems we don't have any locations listed there at the moment.",
                "How about we try a different area? I'm all ears! 👂"
            ],
            [
                f"🧐 Well, this is unexpected! I can't seem to spot any stores in {location}.",
                "🏙️ It looks like we don't have any shops listed there right now.",
                "Shall we explore options in another location? I'm ready when you are! 🚀"
            ]
        ])

def get_store_details(store_name):
    for store in store_locator_data:
        if store_name.lower() in store.get('Location', '').lower():
            return get_random_response([
                [
                    f"🏬 Ah, {store.get('Location')}! Great choice. Here's what you need to know:",
                    f"📍 Address: {store.get('address')}",
                    f"📞 Phone: {store.get('phone')}",
                    f"🕒 Hours: {store.get('hours')}",
                    f"🌐 Website: {store.get('website')}",
                    "Anything else you'd like to know about this store? Just ask! 😊"
                ],
                [
                    f"🌟 Found it! Here's the scoop on {store.get('Location')}:",
                    f"🏠 You'll find them at: {store.get('address')}",
                    f"☎️ Give them a ring: {store.get('phone')}",
                    f"⏰ They're open: {store.get('hours')}",
                    f"🌐 Check them out online: {store.get('website')}",
                    "Need any more details? I'm all ears! 👂"
                ],
                [
                    f"📍 Bingo! I've got the details for {store.get('Location')} right here:",
                    f"🗺️ Pop by: {store.get('address')}",
                    f"📱 Ring them up: {store.get('phone')}",
                    f"🕰️ Store hours: {store.get('hours')}",
                    f"🖥️ Visit their website: {store.get('website')}",
                    "Curious about anything else? Don't be shy, ask away! 😄"
                ]
            ])
    
    return get_random_response([
        [
            f"🤔 Hmm, I'm drawing a blank on '{store_name}'.",
            "Could you double-check the spelling for me?",
            "Or if you'd like, we can try searching for a different store! 🔍"
        ],
        [
            f"😅 Oops! It seems '{store_name}' is playing hide and seek with me.",
            "Want to give it another shot? Maybe try the full store name?",
            "Or we could look up a different store if you prefer! 🌟"
        ],
        [
            f"🧐 Well, this is puzzling! I can't seem to find any details for '{store_name}'.",
            "Mind giving me another try with the store name?",
            "Alternatively, we could search for a different store. What do you think? 🤔"
        ]
    ])

def find_nearest_store(zipcode):
    for store in store_locator_data:
        # Ensure the store's postal code is formatted as a string and matches the extracted zipcode
        if store.get('postal code') == str(zipcode):
            return get_random_response([
                [
                    f"🎉 Jackpot! I've found a store near you!",
                    f"🏬 The closest one to {zipcode} is our {store.get('Location')} branch.",
                    f"📍 You'll find it at: {store.get('address')}",
                    f"📞 Give them a ring at {store.get('phone')} if you need to.",
                    f"🕒 They're open {store.get('hours')}.",
                    "Anything else you'd like to know about this location? 😊"
                ],
                [
                    f"🌟 Great news! There's a store right in your area!",
                    f"🗺️ For zipcode {zipcode}, your nearest stop is {store.get('Location')}.",
                    f"🏠 The address is {store.get('address')}",
                    f"☎️ Need to reach them? Call {store.get('phone')}.",
                    f"⏰ Their hours are {store.get('hours')}.",
                    "Let me know if you need any more details! 👍"
                ],
                [
                    f"🎊 Success! I've pinpointed a store close to you!",
                    f"🏙️ Zipcode {zipcode} is served by our {store.get('Location')} store.",
                    f"🚶‍♀️ Pop by at {store.get('address')}",
                    f"📱 You can contact them on {store.get('phone')}.",
                    f"🕰️ They welcome customers {store.get('hours')}.",
                    "Anything else you're curious about? I'm here to help! 😄"
                ]
            ])
    
    return get_random_response([
        [
            f"😕 Oops! It looks like {zipcode} is playing hide and seek with our stores.",
            "🔍 Could you double-check that zipcode for me?",
            "Or if you'd like, we can try searching with a different one! 🌟"
        ],
        [
            f"🤔 Hmm, I'm having trouble finding a store near {zipcode}.",
            "✏️ Mind giving that zipcode another look to make sure it's correct?",
            "Alternatively, we could try a different zipcode if you'd like! 🚀"
        ],
        [
            f"🧐 Well, this is a head-scratcher! I can't seem to spot any stores near {zipcode}.",
            "🔢 Would you mind confirming that zipcode for me?",
            "Or we could embark on a new search with a different zipcode! What do you say? 🌈"
        ]
    ])

#---------------------------------------------------------------------------------------------------
# track order
def handle_order_tracking(msg):
    order_id = extract_order_id(msg)
    if order_id:
        current_context = None
        return get_track_status(order_id)
    elif is_valid_email(msg):
        orders = get_orders_by_email(msg)
        if orders:
            response = [f"📦 Great news! I found {len(orders)} order(s) associated with the email {msg}:"]
            for i, order in enumerate(orders, 1):
                response.append(f"\n🛍️ Order {i}:\n{format_order_summary(order)}")
            response.append("\n🔍 Which order would you like more details on? Please provide the order ID.")
            current_context = "awaiting_order_id"
            return response
        else:
            current_context = None
            return ["🤔 Hmm, I couldn't find any orders associated with that email address. Could you please double-check the email and try again?"]
        
# Improved track status functions
def extract_order_id(sentence):
    """Extract the order ID from the user's input."""
    pattern = r'\b(ORD\d{3})\b'
    match = re.search(pattern, sentence.upper())
    return match.group(1) if match else None

def is_valid_email(email):
    """Check if the given string is a valid email address."""
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_regex, email))

def get_orders_by_email(email):
    """Get orders associated with the given email address."""
    with open('TrackStatus.json', 'r') as file:
        data = json.load(file)
    
    return [order for order in data['trackorder'] if order['email'].lower() == email.lower()]

def get_track_status(order_id):
    """Get tracking status for a specific order."""
    with open('TrackStatus.json', 'r') as file:
        data = json.load(file)
    
    order = next((order for order in data['trackorder'] if order['orderID'] == order_id), None)
    
    if order:
        status_emoji = {
            "Processing": "🔄",
            "Shipped": "🚚",
            "Delivered": "📦",
            "Cancelled": "❌"
        }.get(order['orderStatus'], "📋")
        
        status_message = f"{status_emoji} Your order (ID: {order_id}) placed by {order['customerName']} is currently {order['orderStatus']}."
        
        return {
            'response': [
                f"📊 Sure thing! Let me fetch that info for you.",
                status_message,
                f"📅 Order Date: {order['orderDate']}",
                "⏳ Would you like to know the expected delivery date?"
            ],
            'order_data': order,
            'next_question': 'delivery_date'
        }
    
    return {
        'response': [
            f"🔍 Oops! I couldn't find any order with ID {order_id}.",
            "🤔 Could you please double-check the order number and try again?"
        ],
        'order_data': None,
        'next_question': None
    }

def handle_order_followup(user_response, order_data, current_question):
    affirmative_responses = {'yes', 'sure', 'okay', 'yep', 'yeah', 'y', 'ok', 'alright', 'certainly', 'absolutely', 'indeed', 'of course'}
    if any(word in user_response.lower().split() for word in affirmative_responses):
        if current_question == 'delivery_date':
            return [
                f"📅 Great! The Expected Delivery is {order_data['expectedDelivery']}.",
                "🚚 We're working hard to get your package to you on time!",
                "🏠 Would you like to know the shipping address?"
            ], 'shipping_address'
        elif current_question == 'shipping_address':
            return [
                f"📍 Certainly! The Shipping Address is {order_data['shippingAddress']}.",
                "📞 Would you like to know the contact number for this parcel?"
            ], 'contact_number'
        elif current_question == 'contact_number':
            return [
                f"☎️ Of course! The Contact Number for this parcel is {order_data['contactNumber']}.",
                "🙋‍♂️ Is there anything else you'd like to know about your order?"
            ], None
    return ["🤔 Is there anything else you'd like to know about this order?"], None

def format_order_summary(order):
    return f"🛍️ Order ID: {order['orderID']}\n📅 Date: {order['orderDate']}\n🚦 Status: {order['orderStatus']}"

# -----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print("Hello! I'm Sam, your friendly shop assistant. How can I help you today? (Type 'quit' to exit)")
        while True:
            sentence = input("You: ")
            if sentence.lower() == "quit":
                print("Sam: Thank you for chatting with me. Have a great day!")
                break
            else:
                resp = get_response(sentence)
                
            if isinstance(resp, list):
                for r in resp:
                    print("Sam:", r)
                    time.sleep(1)  # Add a small delay between responses
            else:
                print("Sam:", resp)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully...")
