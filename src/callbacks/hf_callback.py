from huggingface_hub import HfApi


def push_to_hub_callback(
    model_dir: str, repo_id: str, commit_message: str = "Upload model checkpoint"
):
    """Call this at the end of training or at specific checkpoints to push to HF."""
    print(f"Pushing files from {model_dir} to Hugging Face Hub: {repo_id}...")

    api = HfApi()

    # Creates the repo if it doesn't exist, otherwise does nothing
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    # Uploads the entire folder (weights, configs, tokenizer files)
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    print("Upload complete!")
