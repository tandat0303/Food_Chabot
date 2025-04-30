import json
import re
import inflect
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional
from gensim.models import FastText
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
import logging
import sys
import traceback
import os

logger = logging.getLogger(__name__)

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('search_engine.log', encoding='utf-8')  # Ghi log vào tệp
    ]
)

# Đảm bảo console hỗ trợ UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger.debug("Starting initialization of search_engine.py")

# Cấu hình hằng số từ biến môi trường
CONFIG = {
    "DATA_PATH": os.getenv("DATA_PATH", "data/recipes.json"),
    "FASTTEXT_PATH": os.getenv("FASTTEXT_PATH", "models/fasttext_model.bin"),
    "BERT_PATH": os.getenv("BERT_PATH", "models/bert_model"),
    "SIMILARITY_THRESHOLD": float(os.getenv("SIMILARITY_THRESHOLD", 0.1)),
    "VECTOR_SIZE": int(os.getenv("VECTOR_SIZE", 100)),
    "DQN_LEARNING_RATE": float(os.getenv("DQN_LEARNING_RATE", 0.001)),
    "DQN_GAMMA": float(os.getenv("DQN_GAMMA", 0.9)),
    "DQN_EPSILON": float(os.getenv("DQN_EPSILON", 0.1)),
    "DQN_MEMORY_SIZE": int(os.getenv("DQN_MEMORY_SIZE", 1000)),
    "DQN_BATCH_SIZE": int(os.getenv("DQN_BATCH_SIZE", 32)),
    "DQN_UPDATE_FREQ": int(os.getenv("DQN_UPDATE_FREQ", 10))
}

if not os.path.exists(CONFIG["DATA_PATH"]):
    logger.error(f"File not found: {CONFIG['DATA_PATH']}")
    raise FileNotFoundError(f"File not found: {CONFIG['DATA_PATH']}")
if not os.path.exists(CONFIG["FASTTEXT_PATH"]):
    logger.error(f"File not found: {CONFIG['FASTTEXT_PATH']}")
    raise FileNotFoundError(f"File not found: {CONFIG['FASTTEXT_PATH']}")
if not os.path.exists(CONFIG["BERT_PATH"]):
    logger.error(f"Directory not found: {CONFIG['BERT_PATH']}")
    raise FileNotFoundError(f"Directory not found: {CONFIG['BERT_PATH']}")

# Load dữ liệu món ăn
logger.debug(f"Loading recipes from {CONFIG['DATA_PATH']}")
try:
    with open(CONFIG["DATA_PATH"], "r", encoding="utf-8") as file:
        recipes = json.load(file)
    logger.info(f"Loaded {len(recipes)} recipes")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Failed to load recipes: {str(e)}", exc_info=True)
    raise RuntimeError(f"Failed to load recipes: {str(e)}")

# Load mô hình FastText
logger.debug(f"Loading FastText from {CONFIG['FASTTEXT_PATH']}")
try:
    fasttext_model = FastText.load(CONFIG["FASTTEXT_PATH"])
    logger.info(f"Loaded FastText model, vocabulary size: {len(fasttext_model.wv.index_to_key)}")
except Exception as e:
    logger.error(f"Failed to load FastText model: {str(e)}", exc_info=True)
    raise RuntimeError(f"Failed to load FastText model: {str(e)}")

# Load BERT
logger.debug(f"Loading BERT from {CONFIG['BERT_PATH']}")
try:
    tokenizer = BertTokenizer.from_pretrained(CONFIG["BERT_PATH"])
    bert_model = BertModel.from_pretrained(CONFIG["BERT_PATH"])
    bert_model.eval()
    logger.info("Loaded BERT model")
except Exception as e:
    logger.error(f"Failed to load BERT model: {str(e)}", exc_info=True)
    raise RuntimeError(f"Failed to load BERT model: {str(e)}")

# Mạng nơ-ron cho DQN
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        try:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
        except Exception as e:
            logger.error(f"Error in DQN forward: {str(e)}", exc_info=True)
            raise

# Agent DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = CONFIG["DQN_GAMMA"]
        self.epsilon = CONFIG["DQN_EPSILON"]
        self.learning_rate = CONFIG["DQN_LEARNING_RATE"]
        self.memory = []
        self.batch_size = CONFIG["DQN_BATCH_SIZE"]
        self.update_freq = CONFIG["DQN_UPDATE_FREQ"]

        try:
            self.q_network = DQN(state_size, 128, action_size)
            self.target_network = DQN(state_size, 128, action_size)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()
            self.step_count = 0
            self.update_target_network()
            logger.debug(f"Initialized DQNAgent with state_size={state_size}, action_size={action_size}")
        except Exception as e:
            logger.error(f"Error initializing DQNAgent: {str(e)}", exc_info=True)
            raise

    def update_target_network(self):
        try:
            self.target_network.load_state_dict(self.q_network.state_dict())
        except Exception as e:
            logger.error(f"Error updating target network: {str(e)}", exc_info=True)
            raise

    def get_state(self, user_id: str, recipe: Dict) -> np.ndarray:
        logger.debug(f"Creating state for user_id: {user_id}, recipe: {recipe.get('name', 'No name')}")
        try:
            user_history_vector = np.zeros(10)
            recipe_ingredients = recipe.get("ingredients", [])
            if not recipe_ingredients:
                logger.warning(f"Recipe {recipe.get('name', 'No name')} has no ingredients")
                recipe_vector = np.zeros(CONFIG["VECTOR_SIZE"])
            else:
                ingredient_vectors = [get_fasttext_vector(ing) for ing in recipe_ingredients if normalize_ingredient(ing)]
                recipe_vector = np.mean(ingredient_vectors, axis=0) if ingredient_vectors else np.zeros(CONFIG["VECTOR_SIZE"])

            nutrition_score = self.calculate_nutrition_score(recipe)
            time_score = self.calculate_time_score(recipe)

            state = np.concatenate([user_history_vector, recipe_vector, [nutrition_score, time_score]])
            logger.debug(f"State: {state[:10]}...")
            return state
        except Exception as e:
            logger.error(f"Error in get_state for recipe {recipe.get('name', 'No name')}: {str(e)}", exc_info=True)
            raise

    def calculate_nutrition_score(self, recipe: Dict) -> float:
        try:
            nutrients = recipe.get("nutrients", {})
            if not nutrients or not isinstance(nutrients, dict):
                logger.warning(f"No nutrition data for recipe {recipe.get('name', 'No name')}. Assigning default score 0.")
                return 0

            carbs = parse_float(nutrients.get("carbs", 0))
            protein = parse_float(nutrients.get("protein", 0))
            fat = parse_float(nutrients.get("fat", 0))
            score = (protein * 0.4 + fat * 0.4 - carbs * 0.2) / 100
            logger.debug(f"Nutrition score for {recipe.get('name', 'No name')}: {score}")
            return max(0, score)
        except Exception as e:
            logger.error(f"Error in calculate_nutrition_score for recipe {recipe.get('name', 'No name')}: {str(e)}", exc_info=True)
            raise

    def parse_time(self, time_str: str) -> float:
        """
        Phân tích chuỗi thời gian và trả về số phút dạng float.
        Hỗ trợ các định dạng như: '20 mins', '1 hr and 15 mins', '18 mins - 20 mins'.
        """
        try:
            if not time_str or not isinstance(time_str, str):
                logger.warning(f"Invalid time_str: {time_str}, returning 0")
                return 0

            time_str = time_str.lower().strip()

            # Xử lý khoảng thời gian (e.g., "18 mins - 20 mins")
            if '-' in time_str:
                range_times = time_str.split('-')
                times = [self.parse_time(t.strip()) for t in range_times if t.strip()]
                if times:
                    return sum(times) / len(times)
                return 0

            hr_pattern = r'(\d+)\s*(?:hr|hrs|hour|hours)\b'
            min_pattern = r'(\d+)\s*(?:min|mins|minutes|minute)\b'

            hours = 0
            minutes = 0

            # Tìm giờ
            hr_match = re.search(hr_pattern, time_str)
            if hr_match:
                hours = int(hr_match.group(1))

            # Tìm phút
            min_match = re.search(min_pattern, time_str)
            if min_match:
                minutes = int(min_match.group(1))

            total_minutes = hours * 60 + minutes
            logger.debug(f"Parsed time '{time_str}' to {total_minutes} minutes")
            return float(total_minutes)

        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing time_str '{time_str}': {str(e)}")
            return 0

    def calculate_time_score(self, recipe: Dict) -> float:
        """
        Tính điểm thời gian dựa trên tổng thời gian chuẩn bị và nấu ăn.
        """
        try:
            times = recipe.get("times", {})
            prep_time = self.parse_time(times.get("Preparation", "0 mins"))
            cook_time = self.parse_time(times.get("Cooking", "0 mins"))
            total_time = prep_time + cook_time
            max_reasonable_time = 120  # 2 giờ
            if total_time > max_reasonable_time:
                logger.debug(f"Total time {total_time} exceeds max {max_reasonable_time}, score: 0")
                return 0
            score = max(0, 1 - (total_time / max_reasonable_time))
            logger.debug(f"Time score for {recipe.get('name', 'No name')}: {score} (total_time: {total_time} mins)")
            return score
        except Exception as e:
            logger.error(f"Error in calculate_time_score for recipe {recipe.get('name', 'No name')}: {str(e)}", exc_info=True)
            raise

    def choose_action(self, state: np.ndarray) -> int:
        logger.debug(f"Choosing action with state: {state[:10]}...")
        try:
            if random.random() < self.epsilon:
                action = random.randint(0, self.action_size - 1)
                logger.debug(f"Random action chosen: {action}")
                return action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
                logger.debug(f"Optimal action chosen: {action}, Q-values: {q_values.tolist()[:5]}...")
                return action
        except Exception as e:
            logger.error(f"Error in choose_action: {str(e)}", exc_info=True)
            raise

    def train(self):
        try:
            if len(self.memory) < self.batch_size:
                logger.debug("Memory not sufficient for DQN training")
                return

            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            loss = self.criterion(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step_count += 1
            logger.debug(f"Trained DQN, loss: {loss.item()}, step_count: {self.step_count}")
            if self.step_count % self.update_freq == 0:
                self.update_target_network()
        except Exception as e:
            logger.error(f"Error in DQN train: {str(e)}", exc_info=True)
            raise

    def store_experience(self, state, action, reward, next_state, done):
        try:
            self.memory.append((state, action, reward, next_state, done))
            if len(self.memory) > CONFIG["DQN_MEMORY_SIZE"]:
                self.memory.pop(0)
            logger.debug(f"Stored experience: action={action}, reward={reward}")
        except Exception as e:
            logger.error(f"Error in store_experience: {str(e)}", exc_info=True)
            raise

# Khởi tạo agent
try:
    state_size = CONFIG["VECTOR_SIZE"] + 2
    action_size = len(recipes)
    dqn_agent = DQNAgent(state_size, action_size)
    logger.info("Initialized DQNAgent")
except Exception as e:
    logger.error(f"Failed to initialize DQNAgent: {str(e)}", exc_info=True)
    raise

# Các hàm phụ trợ
def normalize_ingredient(ingredient: str) -> str:
    try:
        if not isinstance(ingredient, str) or not ingredient.strip():
            logger.warning(f"Invalid ingredient: {ingredient}")
            return ""

        ingredient = ingredient.lower().strip()
        ingredient = re.sub(r'\d+', '', ingredient)
        measurement_units = {
            'g', 'gram', 'grams', 'kg', 'kilogram', 'kilograms',
            'ml', 'milliliter', 'milliliters', 'l', 'liter', 'liters',
            'tbsp', 'tablespoon', 'tablespoons', 'tsp', 'teaspoon', 'teaspoons',
            'cup', 'cups', 'can', 'cans', 'bottle', 'bottles',
            'slice', 'slices', 'piece', 'pieces', 'clove', 'cloves',
            'stalk', 'stalks', 'sprig', 'sprigs', 'handful', 'pinch', 'dash'
        }
        ingredient = re.sub(r'\b(' + '|'.join(measurement_units) + r')\b', '', ingredient, flags=re.IGNORECASE)
        stop_words = {
            'cut', 'into', 'chunks', 'wedges', 'deseeded', 'chopped', 'diced', 'sliced',
            'small', 'medium', 'large', 'soft', 'dried', 'ground', 'fresh', 'mild', 'finely', 'rinsed'
        }
        ingredient = re.sub(r'[,:\-–—\s]+', ' ', ingredient)
        words = ingredient.split()
        filtered_words = [word for word in words if word and word not in stop_words]

        if not filtered_words:
            logger.warning(f"No valid words after normalizing: {ingredient}")
            return ""

        p = inflect.engine()
        normalized_words = [p.singular_noun(word) or word for word in filtered_words]
        result = ' '.join(normalized_words)
        logger.debug(f"Normalized ingredient: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in normalize_ingredient for {ingredient}: {str(e)}", exc_info=True)
        return ""

def get_fasttext_vector(ingredient: str) -> np.ndarray:
    try:
        logger.info(f"Processing FastText vector for ingredient: {ingredient}")
        normalized_ing = normalize_ingredient(ingredient)
        logger.debug(f"Normalized ingredient: {normalized_ing}")

        if not normalized_ing:
            logger.warning(f"No valid words after normalizing {ingredient}, returning zero vector")
            return np.zeros(CONFIG["VECTOR_SIZE"])

        words = normalized_ing.split()
        vectors = [fasttext_model.wv[word] for word in words if word in fasttext_model.wv]
        if not vectors:
            logger.warning(f"No vectors found for words in {normalized_ing}, returning zero vector")
            return np.zeros(CONFIG["VECTOR_SIZE"])

        vector = np.mean(vectors, axis=0)
        logger.debug(f"FastText vector: {vector[:5]}...")
        return vector
    except Exception as e:
        logger.error(f"Error in get_fasttext_vector for {ingredient}: {str(e)}", exc_info=True)
        return np.zeros(CONFIG["VECTOR_SIZE"])

def get_bert_embedding(text: str) -> np.ndarray:
    try:
        if not text:
            logger.warning("Empty text for BERT embedding, returning zero vector")
            return np.zeros(768)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy().flatten()
        logger.debug(f"BERT embedding: {embedding[:5]}...")
        return embedding
    except Exception as e:
        logger.error(f"Error in get_bert_embedding: {str(e)}", exc_info=True)
        return np.zeros(768)

def is_vegetarian(recipe: Dict) -> bool:
    try:
        return recipe.get("subcategory", "").lower() == "vegetarian"
    except Exception as e:
        logger.error(f"Error in is_vegetarian: {str(e)}", exc_info=True)
        return False

def is_keto(recipe: Dict) -> bool:
    try:
        nutrients = recipe.get("nutrients", {})
        carbs = parse_float(nutrients.get("carbs", 0))
        return carbs < 20
    except Exception as e:
        logger.error(f"Error in is_keto: {str(e)}", exc_info=True)
        return False

def is_gluten_free(recipe: Dict) -> bool:
    try:
        gluten_ingredients = ["wheat", "flour", "bread", "pasta", "barley", "rye"]
        return all(normalize_ingredient(ing).lower() not in gluten_ingredients for ing in recipe.get("ingredients", []))
    except Exception as e:
        logger.error(f"Error in is_gluten_free: {str(e)}", exc_info=True)
        return False

def parse_float(value):
    try:
        if not value or value == "0":
            return 0.0
        value = re.sub(r'[^\d.]', '', str(value))
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Cannot parse float from value: {value}")
        return 0.0

def categorize_recipe(recipe: Dict) -> str:
    """
    Phân loại công thức thành: appetizer, main, dessert dựa trên tên hoặc nguyên liệu.
    """
    try:
        name = recipe.get("name", "").lower()
        ingredients = [normalize_ingredient(ing).lower() for ing in recipe.get("ingredients", [])]
        total_time = dqn_agent.parse_time(recipe.get("times", {}).get("Preparation", "0 mins")) + \
                     dqn_agent.parse_time(recipe.get("times", {}).get("Cooking", "0 mins"))

        # 从 khóa phân loại
        appetizer_keywords = ["salad", "soup", "starter", "appetizer", "dip"]
        dessert_keywords = ["cake", "cookie", "ice cream", "dessert", "pudding", "custard"]
        main_keywords = ["chicken", "beef", "pork", "fish", "pasta", "rice", "curry"]

        if any(keyword in name for keyword in appetizer_keywords) or \
           (total_time <= 30 and any(ing in ["lettuce", "tomato", "cucumber"] for ing in ingredients)):
            return "appetizer"
        elif any(keyword in name for keyword in dessert_keywords) or \
             any(ing in ["sugar", "chocolate", "cream", "flour"] for ing in ingredients):
            return "dessert"
        else:
            return "main"
    except Exception as e:
        logger.error(f"Error in categorize_recipe for {recipe.get('name', 'No name')}: {str(e)}")
        return "main"

def suggest_meal_plan(user_id: str, ingredient: str = "", diet: str = "all") -> Dict:
    """
    Gợi ý thực đơn với món khai vị, món chính, và món tráng miệng dựa trên chế độ ăn và nguyên liệu.
    """
    logger.info(f"Suggesting meal plan for user_id: {user_id}, ingredient: {ingredient}, diet: {diet}")
    try:
        # Lấy danh sách công thức phù hợp
        matched_recipes = search_recipes_by_ingredient(user_id, ingredient, diet, use_bert=True)
        if not matched_recipes:
            logger.warning("No recipes found for meal plan")
            return {"appetizer": None, "main": None, "dessert": None}

        # Phân loại công thức
        categorized = {"appetizer": [], "main": [], "dessert": []}
        for recipe in matched_recipes:
            category = categorize_recipe(recipe)
            categorized[category].append(recipe)

        meal_plan = {"appetizer": None, "main": None, "dessert": None}
        total_nutrition = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}

        # Chọn công thức tốt nhất cho mỗi danh mục
        for category in meal_plan:
            if categorized[category]:
                # Sử dụng DQN để chọn công thức tốt nhất trong danh mục
                best_recipe = None
                max_reward = float('-inf')
                for recipe in categorized[category]:
                    state = dqn_agent.get_state(user_id, recipe)
                    reward = dqn_agent.calculate_nutrition_score(recipe) + \
                             dqn_agent.calculate_time_score(recipe) + \
                             (user_feedback[user_id].get(recipe["name"], 0) / 5.0)
                    if reward > max_reward:
                        max_reward = reward
                        best_recipe = recipe

                meal_plan[category] = best_recipe
                if best_recipe:
                    nutrients = best_recipe.get("nutrients", {})
                    total_nutrition["calories"] += parse_float(nutrients.get("kcal", 0))
                    total_nutrition["protein"] += parse_float(nutrients.get("protein", 0))
                    total_nutrition["fat"] += parse_float(nutrients.get("fat", 0))
                    total_nutrition["carbs"] += parse_float(nutrients.get("carbs", 0))

        meal_plan_str = ', '.join([f"{k}: {v['name'] if v else 'None'}" for k, v in meal_plan.items() if k != "nutrition_summary"])
        logger.info(f"Generated meal plan: {meal_plan_str}")
        meal_plan["nutrition_summary"] = total_nutrition
        return meal_plan
    except Exception as e:
        logger.error(f"Error in suggest_meal_plan: {str(e)}", exc_info=True)
        return {"appetizer": None, "main": None, "dessert": None, "nutrition_summary": {}}

def search_recipes_by_ingredient(user_id: str, ingredient: str = "", diet: str = "all", use_bert: bool = True) -> List[Dict]:
    logger.info(f"Starting search for user_id: {user_id}, ingredient: {ingredient}, diet: {diet}")
    try:
        query = ingredient.lower().strip() if ingredient else ""
        query_vector = get_fasttext_vector(query) if query else np.zeros(CONFIG["VECTOR_SIZE"])

        if np.all(query_vector == 0) and query:
            logger.warning(f"Query vector for {query} is all zeros, may not exist in model")

        matched_recipes = []
        for i, recipe in enumerate(recipes):
            logger.debug(f"Processing recipe {i}: {recipe.get('name', 'No name')}")
            try:
                normalized_recipe = {
                    "id": recipe.get("id", f"recipe_{i}"),
                    "name": recipe.get("name", "No name"),
                    "image": recipe.get("image", ""),
                    "ingredients": recipe.get("ingredients", []),
                    "subcategory": recipe.get("subcategory", ""),
                    "times": recipe.get("times", {"Preparation": "0 mins", "Cooking": "0 mins"}),
                    "nutrients": recipe.get("nutrients", {}),
                    "steps": recipe.get("steps", []),
                    "avg_rating": user_feedback[user_id].get(recipe.get("name", ""), 0) / 5.0 if user_feedback[user_id].get(recipe.get("name", "")) else 0
                }

                is_match = False
                if diet == "all":
                    is_match = True
                elif diet == "vegetarian":
                    is_match = is_vegetarian(normalized_recipe)
                elif diet == "keto":
                    is_match = is_keto(normalized_recipe)
                elif diet == "gluten_free":
                    is_match = is_gluten_free(normalized_recipe)

                if not is_match:
                    logger.debug(f"Skipping recipe {normalized_recipe['name']} due to diet mismatch: {diet}")
                    continue

                if not query:
                    logger.debug(f"Adding recipe {normalized_recipe['name']} because query is empty")
                    matched_recipes.append((normalized_recipe, 0.0, i))
                    continue

                ingredient_vectors = [get_fasttext_vector(ing) for ing in normalized_recipe["ingredients"] if normalize_ingredient(ing)]
                if not ingredient_vectors:
                    logger.warning(f"No vectors generated for recipe {normalized_recipe['name']}, skipping")
                    continue

                recipe_vector = np.mean(ingredient_vectors, axis=0)
                if np.all(recipe_vector == 0):
                    logger.warning(f"Recipe vector for {normalized_recipe['name']} is all zeros")
                    continue

                try:
                    similarity = cosine_similarity([query_vector], [recipe_vector])[0][0]
                    logger.debug(f"Similarity for {normalized_recipe['name']}: {similarity}")
                except Exception as e:
                    logger.error(f"Error calculating cosine_similarity for {normalized_recipe['name']}: {str(e)}", exc_info=True)
                    continue

                if similarity > CONFIG["SIMILARITY_THRESHOLD"]:
                    logger.debug(f"Adding recipe {normalized_recipe['name']} with similarity {similarity}")
                    matched_recipes.append((normalized_recipe, similarity, i))

            except Exception as e:
                logger.error(f"Error processing recipe {recipe.get('name', 'No name')}: {str(e)}", exc_info=True)
                continue

        logger.info(f"Found {len(matched_recipes)} matching recipes")

        # Kiểm tra matched_recipes trước khi xử lý
        if matched_recipes:
            for item in matched_recipes:
                if not isinstance(item, tuple) or len(item) != 3:
                    logger.error(f"Invalid item in matched_recipes: {item}")
                    raise ValueError(f"Invalid item in matched_recipes: {item}")

        # Tạo result_recipes
        result_recipes = [item[0] for item in matched_recipes]

        # Sử dụng DQN để chọn công thức禁止

        if result_recipes and query:
            logger.debug("Starting DQN selection")
            best_action = -1
            max_reward = float('-inf')
            for j, (recipe_dict, similarity, action_idx) in enumerate(matched_recipes):
                if action_idx >= len(result_recipes):
                    logger.error(f"action_idx {action_idx} exceeds result_recipes length {len(result_recipes)}")
                    continue
                try:
                    state = dqn_agent.get_state(user_id, recipe_dict)
                    reward = dqn_agent.calculate_nutrition_score(recipe_dict) + \
                             dqn_agent.calculate_time_score(recipe_dict) + \
                             (user_feedback[user_id].get(recipe_dict["name"], 0) / 5.0)
                    logger.debug(f"Reward for recipe {recipe_dict['name']}: {reward}")

                    if reward > max_reward:
                        max_reward = reward
                        best_action = action_idx
                except Exception as e:
                    logger.error(f"Error calculating reward for recipe {recipe_dict['name']}: {str(e)}", exc_info=True)
                    continue

            if best_action != -1:
                best_recipe = result_recipes[best_action]
                next_state = dqn_agent.get_state(user_id, best_recipe)
                dqn_agent.store_experience(
                    dqn_agent.get_state(user_id, result_recipes[0]),
                    best_action,
                    max_reward,
                    next_state,
                    done=False
                )
                dqn_agent.train()
                logger.debug(f"Selected best recipe: {best_recipe['name']}")
                return [best_recipe] + [r for i, r in enumerate(result_recipes) if i != best_action]
            else:
                logger.warning("No valid best_action found")

        logger.info(f"Returning {len(result_recipes)} recipes")
        return result_recipes

    except Exception as e:
        logger.error(f"Error in search_recipes_by_ingredient: {str(e)}", exc_info=True)
        raise

def update_user_feedback(user_id: str, recipe_name: str, rating: int) -> None:
    try:
        if not (1 <= rating <= 5):
            raise ValueError("Rating must be from 1 to 5")
        user_feedback[user_id][recipe_name] = rating
        logger.info(f"Updated feedback: user_id={user_id}, recipe={recipe_name}, rating={rating}")
    except Exception as e:
        logger.error(f"Error in update_user_feedback: {str(e)}", exc_info=True)
        raise

# Lưu trữ lịch sử và phản hồi
user_history = defaultdict(list)
user_feedback = defaultdict(lambda: defaultdict(int))

if __name__ == "__main__":
    user_id = "user123"
    user_query = "potato"

    try:
        results = search_recipes_by_ingredient(user_id, user_query, use_bert=True)
        print("Suggested recipes:")
        for recipe in results[:5]:
            print(f"- {recipe['name']}")
    except Exception as e:
        print(f"Search error: {str(e)}")