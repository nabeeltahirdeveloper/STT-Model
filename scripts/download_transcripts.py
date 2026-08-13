import argparse

from huggingface_hub import HfApi, hf_hub_download


def download_files(repo_id: str, files: list[str], local_dir: str):
    print(f"Found {len(files)} files to download.")
    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] Downloading {f}")
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f, local_dir=local_dir)
    print("Download complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-transcripts",
        action="store_true",
        help="Download corpus jsonl transcripts",
    )
    parser.add_argument("--benchmark", action="store_true", help="Download benchmark files")
    args = parser.parse_args()

    api = HfApi()
    repo_id = "ASLP-lab/UrduSpeech"
    print(f"Listing files in {repo_id}...")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    to_download = []
    if args.corpus_transcripts:
        to_download.extend(
            f for f in files if f.endswith("_final_transcription.jsonl") and f.startswith("corpus/")
        )

    if args.benchmark:
        to_download.extend(f for f in files if f.startswith("benchmark/"))

    if not to_download:
        print("Nothing to download. Please specify --corpus-transcripts or --benchmark.")
        return

    download_files(repo_id, to_download, "data/raw/urduspeech")


if __name__ == "__main__":
    main()
