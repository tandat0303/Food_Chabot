import json
import os
import re
from typing import List, Dict
from pathlib import Path
import inflect
import torch
from transformers import BertTokenizer, BertModel
from gensim.models import FastText

# Cấu hình hằng số
CONFIG = {
    "DATA_PATH": "data/recipes.json",
    "MODEL_DIR": "models",
    "FASTTEXT_PATH": "models/fasttext_model.bin",
    "BERT_PATH": "models/bert_model",
    "VECTOR_SIZE": 100,
    "WINDOW": 5,
    "MIN_COUNT": 1,
    "WORKERS": 4,
    "BERT_MODEL_NAME": "bert-base-uncased"
}

def ensure_directory_exists(path: str) -> None:
    """Đảm bảo thư mục tồn tại, nếu không thì tạo mới."""
    Path(path).mkdir(parents=True, exist_ok=True)

def load_recipes(file_path: str) -> List[Dict]:
    """Load dữ liệu món ăn từ file JSON với xử lý lỗi."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Dữ liệu JSON không hợp lệ trong file: {file_path}")

def preprocess_ingredient(ingredient: str) -> List[str]:
    """Chuẩn hóa nguyên liệu, trả về danh sách các từ chính."""
    if not isinstance(ingredient, str) or not ingredient.strip():
        return []

    ingredient = ingredient.lower().strip()

    # Loại bỏ số và đơn vị đo lường
    ingredient = re.sub(r'\d+', '', ingredient)  # Bỏ số
    measurement_units = {
        'g', 'gram', 'grams', 'kg', 'kilogram', 'kilograms',
        'ml', 'milliliter', 'milliliters', 'l', 'liter', 'liters',
        'tbsp', 'tablespoon', 'tablespoons', 'tsp', 'teaspoon', 'teaspoons',
        'cup', 'cups', 'can', 'cans', 'bottle', 'bottles',
        'slice', 'slices', 'piece', 'pieces', 'clove', 'cloves',
        'stalk', 'stalks', 'sprig', 'sprigs', 'handful', 'pinch', 'dash'
    }
    ingredient = re.sub(r'\b(' + '|'.join(measurement_units) + r')\b', '', ingredient, flags=re.IGNORECASE)

    # Loại bỏ các từ mô tả không cần thiết
    stop_words = {
        'cut', 'into', 'chunks', 'wedges', 'deseeded', 'chopped', 'diced', 'sliced',
        'small', 'medium', 'large', 'soft', 'dried', 'ground', 'fresh', 'mild'
    }
    ingredient = re.sub(r'[,:\-–—\s]+', ' ', ingredient)  # Thay dấu câu và khoảng trắng thừa bằng một khoảng trắng
    words = ingredient.split()
    filtered_words = [word for word in words if word and word not in stop_words]

    # Chuyển danh từ về số ít
    p = inflect.engine()
    normalized_words = []
    for word in filtered_words:
        singular = p.singular_noun(word) or word
        normalized_words.append(singular)

    # Giữ các cụm từ quan trọng (như "olive oil", "clotted cream")
    return normalized_words if normalized_words else []

def train_models(recipes: List[Dict]) -> None:
    """Huấn luyện mô hình FastText và tải BERT, sau đó lưu lại."""
    unique_ingredients: set = set()
    processed_ingredient_sentences: list[list[str]] = []

    # Xử lý nguyên liệu
    for recipe in recipes:
        processed_ingredients = []
        for ing in recipe.get("ingredients", []):
            clean_ings = preprocess_ingredient(ing)
            for clean_ing in clean_ings:
                if clean_ing and clean_ing not in unique_ingredients:
                    unique_ingredients.add(clean_ing)
                    processed_ingredients.append(clean_ing)

        if processed_ingredients:
            processed_ingredient_sentences.append(processed_ingredients)

    if not processed_ingredient_sentences:
        raise ValueError("Không có nguyên liệu nào được xử lý.")

    # Huấn luyện FastText
    try:
        fasttext_model = FastText(
            sentences=processed_ingredient_sentences,
            vector_size=CONFIG["VECTOR_SIZE"],
            window=CONFIG["WINDOW"],
            min_count=CONFIG["MIN_COUNT"],
            workers=CONFIG["WORKERS"]
        )
        fasttext_model.save(CONFIG["FASTTEXT_PATH"])
        print("Đã huấn luyện và lưu mô hình FastText thành công!")
    except Exception as e:
        raise RuntimeError(f"Lỗi khi huấn luyện FastText: {str(e)}")

    # Tải và lưu BERT
    try:
        tokenizer = BertTokenizer.from_pretrained(CONFIG["BERT_MODEL_NAME"])
        bert_model = BertModel.from_pretrained(CONFIG["BERT_MODEL_NAME"])
        ensure_directory_exists(CONFIG["BERT_PATH"])
        bert_model.save_pretrained(CONFIG["BERT_PATH"])
        tokenizer.save_pretrained(CONFIG["BERT_PATH"])
        print("Đã tải và lưu mô hình BERT thành công!")
    except Exception as e:
        raise RuntimeError(f"Lỗi khi tải hoặc lưu BERT: {str(e)}")

def main() -> None:
    """Hàm chính để chạy toàn bộ quá trình."""
    ensure_directory_exists(CONFIG["MODEL_DIR"])

    try:
        recipes = load_recipes(CONFIG["DATA_PATH"])
        train_models(recipes)
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        raise

if __name__ == "__main__":
    main()