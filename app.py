from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from models.search_engine import search_recipes_by_ingredient, update_user_feedback, suggest_meal_plan
import logging
from typing import Dict, Any
from werkzeug.exceptions import BadRequest
import traceback
import os

# Cấu hình ứng dụng
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "food-chatbot.vercel.app"}})

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Đảm bảo console hỗ trợ UTF-8
import sys
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Hằng số
VALID_RATING_RANGE = range(1, 6)

def validate_json(data: Dict[str, Any], required_fields: set) -> None:
    if not data or not all(field in data for field in required_fields):
        raise BadRequest(f"Missing required fields: {', '.join(required_fields)}")

@app.route("/")
def home() -> str:
    logger.debug("Accessing home page")
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search() -> tuple[Dict[str, Any], int]:
    try:
        data = request.get_json(silent=True)
        logger.debug("Received data from frontend: %s", data)

        validate_json(data, {"user_id"})
        user_id = data["user_id"].strip()
        ingredient = data.get("ingredients", "").strip()
        diet = data.get("diet", "all").strip()

        if not user_id:
            raise BadRequest("user_id cannot be empty")

        valid_diets = {"all", "vegetarian", "keto", "gluten_free"}
        if diet not in valid_diets:
            raise BadRequest(f"Invalid diet: {diet}")

        logger.info("Searching for user %s with ingredient: %s, diet: %s", user_id, ingredient, diet)

        results = search_recipes_by_ingredient(user_id, ingredient, diet)
        logger.debug("Search results: %d recipes", len(results))

        if not results:
            return jsonify({"message": "No recipes found", "results": []}), 200

        return jsonify(results), 200

    except BadRequest as e:
        logger.warning("Input error: %s", str(e))
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Server error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}", "traceback": traceback.format_exc()}), 500

@app.route('/meal_plan', methods=['POST'])
def meal_plan():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        ingredient = data.get('ingredients', '')
        diet = data.get('diet', 'all')

        meal_plan = suggest_meal_plan(user_id, ingredient, diet)
        return jsonify(meal_plan)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/feedback", methods=["POST"])
def feedback() -> tuple[Dict[str, Any], int]:
    try:
        data = request.get_json(silent=True)
        logger.debug("Received feedback data from frontend: %s", data)

        validate_json(data, {"user_id", "recipe_name", "rating"})
        user_id = data["user_id"].strip()
        recipe_name = data["recipe_name"].strip()
        rating = data["rating"]

        if not user_id or not recipe_name:
            raise BadRequest("user_id and recipe_name cannot be empty")

        try:
            rating = int(rating)
            if rating not in VALID_RATING_RANGE:
                raise ValueError
        except ValueError:
            raise BadRequest("Rating must be an integer from 1 to 5")

        update_user_feedback(user_id, recipe_name, rating)
        logger.info("User %s rated recipe %s: %d stars", user_id, recipe_name, rating)
        return jsonify({"message": "Feedback recorded!"}), 200

    except BadRequest as e:
        logger.warning("Input error: %s", str(e))
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error("Server error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}", "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    logger.info("Starting Flask server in development mode...")
    app_config = {
        "debug": True,
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 5000))
    }
    app.run(**app_config)