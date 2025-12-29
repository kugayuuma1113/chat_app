from huggingface_hub import hf_hub_download

print("モデルをダウンロード中...（数GBあるので時間がかかります）")
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    filename="qwen2.5-3b-instruct-q4_k_m.gguf",
    local_dir="./models" # プロジェクト内のmodelsフォルダに保存
)
print(f"完了！保存先: {model_path}")