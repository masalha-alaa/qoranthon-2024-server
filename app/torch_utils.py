import os.path

from root import *
import zipfile
from pyarabic.araby import strip_tashkeel, normalize_hamza
from sentence_transformers import SentenceTransformer
import torch
import boto3


class ModelLoading:
    SEED = 42

    def __init__(self, model_name):
        self._models_dir = "model"
        self.files_path = ROOT_DIR / self._models_dir
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
        self.model_name = model_name
        self.model_path = self.files_path / self.model_name

    def initialize(self, overwrite, environment, clear_after_extract=True):
        print("Initializing...")
        if not os.path.exists(self._models_dir):
            print(f"Creating dir:\n{self._models_dir}")
            os.mkdir(self._models_dir)
        else:
            print("Dir already exists")

        model_zip_file_name = f"{self.model_name}.zip"
        path_to_zip_file = os.path.join(self._models_dir, model_zip_file_name)
        if environment == "s3":
            print("Connecting to s3...")
            s3 = boto3.client('s3')
            if (not os.path.exists(path_to_zip_file) and not os.path.exists(os.path.join(self._models_dir, self.model_name))) or overwrite:
                print(f"Downloading {model_zip_file_name} to {path_to_zip_file}")
                s3.download_file('my-dl-models-heroku', model_zip_file_name, path_to_zip_file)
                print("Done downloading")
            else:
                print("Zip file already exists")
        else:  # local
            if not os.path.exists(path_to_zip_file):
                raise FileNotFoundError(f"Model not found at {path_to_zip_file}")
        if not os.path.exists(os.path.join(self._models_dir, self.model_name)) or overwrite:
            print("Extracting zip...")
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(self._models_dir)
            print("Done extracting")
            if os.path.exists(path_to_zip_file) and clear_after_extract:
                print("Deleting zip file...")
                os.remove(path_to_zip_file)
                print("Done deleting")
        else:
            print("Model already extracted")

        print("Loading model...")
        self._load_model(self.device)
        print("Done loading model")

        return "OK"

    def clean_line(self, line):
        # normalize
        cleaned = strip_tashkeel(line)
        cleaned = normalize_hamza(cleaned, method='tasheel')
        return cleaned

    def _load_model(self, device):
        sentence_transformer_path = str(self.model_path)
        print(f"Initializing from:\n{sentence_transformer_path}")
        self.model = None
        self.model = SentenceTransformer(sentence_transformer_path, device=device)

    def predict(self, sentences):
        if self.model is not None:
            encoding = self.model.encode(self.clean_line(sentences))
            return encoding.tolist()
        print("Model is not initialized")
        return []


# if __name__ == '__main__':
#     print("Encoding...")
#     model_loading = ModelLoading()
#     print(model_loading.predict(['slim shady', 'i been crazy']))
