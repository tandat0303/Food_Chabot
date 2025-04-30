# Food Chatbot (Search Recipe App)
This is a Flask-based web application that allows users to search for recipes based on ingredients, dietary preferences, and nutritional requirements. The backend uses BERT and FastText models for semantic search and meal plan suggestions, while the frontend provides an interactive interface for recipe searching, meal planning, and user ratings. The application is designed to be deployed on Render with large files (e.g., recipes.json, BERT/FastText models) hosted on Google Drive.

## Author

- [@tandat0303](https://www.github.com/tandat0303)

## Features
- Recipe Search: Search for recipes by ingredients and dietary preferences (e.g., vegetarian, keto, gluten-free).
- Meal Plan Suggestions: Generate meal plans with appetizer, main course, and dessert based on user input.
- Nutritional Evaluation: Calculate Dietary Quality Index-International (DQI-I) scores for recipes.
- User Ratings: Allow users to rate recipes and view average ratings.
- Search History: Store and display recent search queries.
- Responsive Frontend: Built with HTML, CSS, and JavaScript for a user-friendly experience.

## Tech Stack
- Backend: Flask, Python, BERT (Transformers), FastText
- Frontend: HTML, CSS, JavaScript (with Bootstrap for styling)
- Storage: IndexedDB for caching, localStorage for user data
- Deployment: Render
- External Storage: Google Drive for large files (recipes.json, BERT/FastText models)

## Project Structure
```plain
Food_Chatbot/
├── app.py                    # Flask application
├── models/
│   ├── search_engine.py     # Search and meal plan logic with BERT/FastText
│   ├── train_model.py       # Model training logic for BERT/FastText
├── download_models.py        # Script to download large files during deployment
├── static/
│   ├── script.js            # Frontend JavaScript logic
│   ├── styles.css           # Custom CSS
│   └── ...
├── templates/
│   ├── index.html           # Main HTML template
│   └── ...
├── Procfile                 # Render deployment configuration
├── requirements.txt         # Python dependencies
└── .gitignore               # Git ignore file
```
## Prerequisites
- Python 3.8+
- Git
- Render account for backend deployment
- Vercel account for frontend deployment
- Google Drive for hosting large files
- Postman (optional, for API testing)

## Setup and Local Development
1. Clone the repository
```bash
git clone https://github.com/tandat0303/Food_Chatbot.git
cd Food_Chatbot
```

2. Install Dependencies
Create a virtual environment and install Python dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. Prepare Large Files
The application requires large files (recipes.json, BERT/FastText models) that are not included in the repository due to GitHub size limits. These files are downloaded during deployment, but for local development, you need to place them manually:
- Download Files:
    - data/recipes.json: Recipe dataset
    - models/bert_model/: BERT model directory
    - models/fasttext_model.bin: FastText model
    - models/fasttext_model.bin.wv.vectors_ngrams.npy: FastText n-grams
    - Contact the repository owner for access to these files or host them on     Google Drive.

- Place Files:
    - Create data/ and models/ directories if they don't exist:
    ```bash
    mkdir -p data models
    ```
    - Move recipes.json to data/recipes.json.
    - Move bert_model/, fasttext_model.bin, and fasttext_model.bin.wv.vectors_ngrams.npy to models/.

4. Run the Application Locally
Start the Flask server:
```bash
python app.py
```

- The backend will run at http://127.0.0.1:5000.
- Open http://127.0.0.1:5000 in a browser to access the frontend.

5. Test API Endpoints
Use Postman or curl to test the API:
- Search Recipes:
```bash
curl -X POST http://127.0.0.1:5000/search \
-H "Content-Type: application/json" \
-d '{"user_id": "user123", "ingredients": "potato", "diet": "all"}'
```

- Meal Plan:
```bash
curl -X POST http://127.0.0.1:5000/meal_plan \
-H "Content-Type: application/json" \
-d '{"user_id": "user123", "ingredients": "potato", "diet": "all"}'
```

- Feedback:
```bash
curl -X POST http://127.0.0.1:5000/feedback \
-H "Content-Type: application/json" \
-d '{"user_id": "user123", "recipe_name": "Potato Soup", "rating": 5}'
```

## Deployment
Backend Deployment on Render
1. Host Large Files on Google Drive
- Files to Host:
    - recipes.json (or recipes.json.zip if compressed)
    - bert_model.zip (compressed models/bert_model/)
    - fasttext_model.bin
    - fasttext_model.bin.wv.vectors_ngrams.npy
- Steps:
    - Upload each file to Google Drive.
    - Set sharing to Anyone with the link.
    - Get the direct download URL:
    ```
    https://drive.google.com/uc?export=download&id=FILE_ID
    ```
    - Note down the URLs for use in environment variables.

2. Push Code to GitHub
- Ensure .gitignore excludes large files:
```plain
data/recipes.json
data/recipes.json.zip
models/bert_model/
models/bert_model.zip
models/fasttext_model.bin
models/fasttext_model.bin.wv.vectors_ngrams.npy
*.pyc
__pycache__/
.env
```

- Push code to GitHub:
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

3. Create a Web Service on Render
- Log in to Render Dashboard.
- Click New > Web Service and connect your GitHub repository.
- Configure the service:
    - Name: recipe-app
    - Environment: Python
    - Region: Choose the closest region (e.g., Oregon)
    - Build Command:
    ```plain
    pip install -r requirements.txt
    ```
    - Start Command:
    ```plain
    python download_models.py && gunicorn app:app
    ```
    - Instance Type: Free (or upgrade for more resources)
    - Environment Variables:
    ```
    BERT_MODEL_URL=https://drive.google.com/uc?export=download&    id=YOUR_BERT_FILE_ID
    FASTTEXT_MODEL_URL=https://drive.google.com/uc?export=download&id=YOUR_FASTTEXT_FILE_ID
    FASTTEXT_NGRAMS_URL=https://drive.google.com/uc?export=download&id=YOUR_NGRAMS_FILE_ID
    RECIPES_URL=https://drive.google.com/uc?export=download&id=YOUR_RECIPES_FILE_ID
    ```
    - Click Create Web Service to deploy.

4. Verify Backend Deployment
- Check the Logs tab in Render for download status:
```plain
Đang tải https://drive.google.com/uc?export=download&id=... về data/recipes.json
Tải thành công data/recipes.json
```
- Test API endpoints with Postman:
```bash
curl -X POST https://recipe-app.onrender.com/search \
-H "Content-Type: application/json" \
-d '{"user_id": "user123", "ingredients": "potato", "diet": "all"}'
```
- Note: The free Render tier may have a ~30-second delay when the app wakes up.

    

## Frontend Deployment on Vercel
1. Prepare Frontend
- The frontend files are in static/ (script.js, styles.css) and templates/ (index.html).
- Update script.js with the Render backend URL:
```javascript
const API_CONFIG = {
    BASE_URL: window.location.hostname === "localhost" ? "http://127.0.0.1:5000" : "https://recipe-app.onrender.com",
    // ...
};
```

2. Push Frontend to GitHub
- If the frontend is in the same repository, ensure static/ and templates/ are included.
- Alternatively, create a separate repository for the frontend:
```bash
mkdir recipe-app-frontend
cp -r static templates recipe-app-frontend/
cd recipe-app-frontend
git init
git add .
git commit -m "Initial frontend commit"
git remote add origin https://github.com/your-username/recipe-app-frontend.git
git push origin main
```

3. Deploy on Vercel
- Log in to Vercel.
- Click New Project and connect to the frontend repository.
- Configure:
    - Framework Preset: Other
    - Root Directory: . (or frontend/ if in a subdirectory)
- Deploy and note the URL (e.g., https://recipe-app-frontend.vercel.app).

4. Update CORS in Backend
- Update app.py to allow the Vercel frontend domain:
```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["https://food-chatbot-frontend.vercel.app"]}})
```

5. Verify Frontend
- Open the Vercel URL in a browser.
- Test recipe search, meal planning, and rating features.
- Check the browser console (F12 > Console) for CORS or API errors.
## Troubleshooting
- Download Errors:
    - Issue: download_models.py fails to download files.
    - Solution: Verify Google Drive URLs are public and correct. Check Render logs for details.
- JSON Parsing Errors:
    - Issue: recipes.json causes errors in models/search_engine.py.
    - Solution: Validate JSON format:
    ```bash
    jq . data/recipes.json
    ```
- Disk Space Errors:
    - Issue: Render free tier runs out of disk space.
    - Solution:
        - Optimize recipes.json (remove unused fields, store images as URLs).
        - Use lighter models (e.g., distilbert-base-uncased for BERT).
        - Upgrade to a paid Render plan.
- CORS Errors:
    - Issue: Frontend reports CORS errors.
    - Solution: Ensure app.py allows the Vercel domain in CORS settings.
- API Latency:
    - Issue: Render free tier has ~30-second wake-up delay.
    - Solution: script.js includes retry logic (fetchWithRetry). Consider a paid plan for better performance.
## Optimization Suggestions 
- Reduce File Sizes:
    - Optimize recipes.json by removing unnecessary fields or using a database (e.g., MongoDB Atlas).
    - Use lighter BERT models (e.g., distilbert-base-uncased).
- Database Integration:
    - Migrate recipes.json to MongoDB or PostgreSQL for faster queries:
    ```python
    from pymongo import MongoClient
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client["recipe_app"]
    recipes = list(db.recipes.find())
    ```
- Caching:
    - Enhance backend caching (e.g., Redis) to reduce API response time.
- Security:
    - Use AWS S3 with presigned URLs instead of Google Drive for sensitive files.


## Contributing
- Fork the repository.
- Create a feature branch (git checkout -b feature/your-feature).
- Commit changes (git commit -m "Add your feature").
- Push to the branch (git push origin feature/your-feature).
- Open a Pull Request.


## License
This project is licensed under the [MIT License.](https://choosealicense.com/licenses/mit/)


## Contact
For issues or questions, open an issue on GitHub or contact [dat.truongtan03@gmail.com].