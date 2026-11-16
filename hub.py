"""Download / upload the trained weights from / to the Hugging Face Hub
(weights are not stored in the git repository)."""
from .config import HF_REPO_ID, MODELS_DIR, ONNX_INT8


def ensure_models(include_onnx_fp32: bool = False):
    """Download the weights into ./models if they are not there yet."""
    if ONNX_INT8.exists():
        return
    if HF_REPO_ID.startswith("HF_USERNAME/"):
        raise SystemExit("Set HF_REPO_ID (see README) or train the models first.")
    from huggingface_hub import snapshot_download
    print(f"Downloading models from https://huggingface.co/{HF_REPO_ID} ...")
    ignore = [] if include_onnx_fp32 else ["berturk_onnx/model_fp32.onnx"]
    snapshot_download(repo_id=HF_REPO_ID, local_dir=str(MODELS_DIR), ignore_patterns=ignore)


def upload_models():
    from huggingface_hub import create_repo, upload_folder
    create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
    upload_folder(repo_id=HF_REPO_ID, folder_path=str(MODELS_DIR), repo_type="model",
                  commit_message="Upload trained models")
    print(f"Uploaded to https://huggingface.co/{HF_REPO_ID}")
