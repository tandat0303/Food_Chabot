import os
import urllib.request
import zipfile
import shutil
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, dest):
    """Tải tệp từ URL về đích."""
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        logger.info(f"Đang tải {url} về {dest}")
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Tải thành công {dest}")
    except Exception as e:
        logger.error(f"Lỗi khi tải {url}: {str(e)}")
        raise

def unzip_file(zip_path, extract_to):
    """Giải nén tệp zip."""
    try:
        logger.info(f"Giải nén {zip_path} vào {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Giải nén thành công vào {extract_to}")
    except Exception as e:
        logger.error(f"Lỗi khi giải nén {zip_path}: {str(e)}")
        raise

def download_models():
    """Tải và cấu hình các mô hình và dữ liệu."""
    try:
        # Định nghĩa các URL và đích
        model_files = [
            {
                "url": os.getenv("BERT_MODEL_URL", "https://drive.google.com/uc?export=download&id=YOUR_BERT_FILE_ID"),
                "dest": "models/bert_model.zip",
                "extract_to": "models/bert_model"
            },
            {
                "url": os.getenv("FASTTEXT_MODEL_URL", "https://drive.google.com/uc?export=download&id=YOUR_FASTTEXT_FILE_ID"),
                "dest": "models/fasttext_model.bin"
            },
            {
                "url": os.getenv("FASTTEXT_NGRAMS_URL", "https://drive.google.com/uc?export=download&id=YOUR_NGRAMS_FILE_ID"),
                "dest": "models/fasttext_model.bin.wv.vectors_ngrams.npy"
            },
            {
                "url": os.getenv("RECIPES_URL", "https://drive.google.com/uc?export=download&id=YOUR_RECIPES_FILE_ID"),
                "dest": "data/recipes.json"
            }
        ]

        for model in model_files:
            download_file(model["url"], model["dest"])
            if model.get("extract_to"):
                unzip_file(model["dest"], model["extract_to"])
                os.remove(model["dest"])

        logger.info("Hoàn tất tải và cấu hình các mô hình và dữ liệu")
    except Exception as e:
        logger.error(f"Lỗi trong download_models: {str(e)}")
        raise

if __name__ == "__main__":
    download_models()