import os
import urllib.request
import zipfile
import tarfile
import logging
import re
import gdown

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_url(url):
    """Kiểm tra xem URL có hợp lệ không."""
    if not url or not isinstance(url, str):
        return False
    if not re.match(r'^https?://', url):
        return False
    if 'YOUR_' in url:
        return False
    return True

def is_zip_file(filepath):
    """Kiểm tra xem tệp có phải là ZIP không."""
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.testzip()
        return True
    except zipfile.BadZipFile:
        return False

def is_tar_file(filepath):
    """Kiểm tra xem tệp có phải là tar.gz không."""
    return tarfile.is_tarfile(filepath)

def download_file(url, dest):
    """Tải tệp từ URL và lưu vào đích, sử dụng gdown cho Google Drive."""
    try:
        if not validate_url(url):
            raise ValueError(f"URL không hợp lệ: {url}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        logger.info(f"Đang tải {url} về {dest}")
        if 'drive.google.com' in url:
            gdown.download(url, dest, quiet=False)
        else:
            urllib.request.urlretrieve(url, dest)
        logger.info(f"Tải thành công {dest}")
    except Exception as e:
        logger.error(f"Lỗi khi tải {url}: {str(e)}")
        raise

def extract_file(filepath, extract_to):
    """Giải nén tệp ZIP hoặc tar.gz."""
    try:
        os.makedirs(extract_to, exist_ok=True)
        if is_zip_file(filepath):
            logger.info(f"Giải nén ZIP {filepath} vào {extract_to}")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif is_tar_file(filepath):
            logger.info(f"Giải nén tar.gz {filepath} vào {extract_to}")
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Tệp {filepath} không phải ZIP hoặc tar.gz")
        logger.info(f"Giải nén thành công vào {extract_to}")
    except Exception as e:
        logger.error(f"Lỗi khi giải nén {filepath}: {str(e)}")
        raise

def download_models():
    """Tải và cấu hình các mô hình và dữ liệu."""
    try:
        model_files = [
            {
                "url": os.getenv("BERT_MODEL_URL"),
                "dest": "models/bert_model.zip",
                "extract_to": "models/bert_model"
            },
            {
                "url": os.getenv("FASTTEXT_MODEL_URL"),
                "dest": "models/fasttext_model.bin"
            },
            {
                "url": os.getenv("FASTTEXT_NGRAMS_URL"),
                "dest": "models/fasttext_model.bin.wv.vectors_ngrams.npy"
            },
            {
                "url": os.getenv("RECIPES_URL"),
                "dest": "data/recipes.json"
            }
        ]

        for model in model_files:
            if not model["url"]:
                logger.error(f"Biến môi trường cho {model['dest']} không được đặt")
                raise ValueError(f"Biến môi trường cho {model['dest']} không được đặt")
            download_file(model["url"], model["dest"])
            if model.get("extract_to"):
                extract_file(model["dest"], model["extract_to"])
                os.remove(model["dest"])

        logger.info("Hoàn tất tải và cấu hình các mô hình và dữ liệu")
    except Exception as e:
        logger.error(f"Lỗi trong download_models: {str(e)}")
        raise

if __name__ == "__main__":
    download_models()