"""
Auto-download LaMa pretrained model
Supports multiple sources: Hugging Face, Google Drive, Direct URL
"""

import os
import requests
from tqdm import tqdm
import gdown
import torch
from pathlib import Path


class ModelDownloader:
    """Download and setup LaMa model"""

    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.checkpoint_dir / "lama_model.pth"

    def download_from_url(self, url, filename=None):
        """Download file from direct URL with progress bar"""
        if filename is None:
            filename = self.model_path

        print(f"📥 Downloading from: {url}")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as f, tqdm(
            desc=str(filename),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

        print(f"✅ Downloaded to: {filename}")
        return filename

    def download_from_gdrive(self, file_id):
        """Download from Google Drive"""
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"📥 Downloading from Google Drive: {file_id}")

        output = str(self.model_path)
        gdown.download(url, output, quiet=False)

        print(f"✅ Downloaded to: {output}")
        return output

    def download_from_huggingface(self, repo_id="LaMa/lama-main", filename="best.ckpt"):
        """Download from Hugging Face Hub"""
        try:
            from huggingface_hub import hf_hub_download

            print(f"📥 Downloading from Hugging Face: {repo_id}/{filename}")

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.checkpoint_dir)
            )

            # Copy to expected location
            import shutil
            shutil.copy(downloaded_path, self.model_path)

            print(f"✅ Downloaded to: {self.model_path}")
            return str(self.model_path)

        except ImportError:
            print("⚠️  huggingface_hub not installed. Installing...")
            os.system("pip install huggingface_hub")
            return self.download_from_huggingface(repo_id, filename)

    def download_official_lama(self):
        """Download official LaMa model from multiple sources"""

        # Try multiple sources
        sources = [
            {
                "name": "Hugging Face (Recommended)",
                "method": "huggingface",
                "args": {"repo_id": "smarterwallet/lama-models", "filename": "big-lama.pt"}
            },
            {
                "name": "Google Drive Mirror",
                "method": "gdrive",
                "args": {"file_id": "1-5kgXECR8VfQqfvNvXPWNJxc9SZBpDtK"}  # Example ID
            },
            {
                "name": "Direct URL",
                "method": "url",
                "args": {"url": "https://github.com/advimman/lama/releases/download/0.0.0/big-lama.zip"}
            }
        ]

        for source in sources:
            try:
                print(f"\n🔄 Trying source: {source['name']}")

                if source['method'] == 'huggingface':
                    return self.download_from_huggingface(**source['args'])
                elif source['method'] == 'gdrive':
                    return self.download_from_gdrive(**source['args'])
                elif source['method'] == 'url':
                    downloaded = self.download_from_url(**source['args'])

                    # If it's a zip file, extract it
                    if str(downloaded).endswith('.zip'):
                        import zipfile
                        print("📦 Extracting zip file...")
                        with zipfile.ZipFile(downloaded, 'r') as zip_ref:
                            zip_ref.extractall(self.checkpoint_dir)
                        os.remove(downloaded)
                        print("✅ Extraction complete")

                    return downloaded

            except Exception as e:
                print(f"❌ Failed: {e}")
                print("   Trying next source...\n")
                continue

        print("\n⚠️  All download sources failed!")
        print("📝 Manual download instructions:")
        print("   1. Visit: https://github.com/advimman/lama/releases")
        print("   2. Download 'big-lama.zip'")
        print(f"   3. Extract and place 'big-lama.pt' in: {self.checkpoint_dir}")
        print(f"   4. Rename to: lama_model.pth")

        return None

    def verify_model(self):
        """Verify downloaded model can be loaded"""
        if not self.model_path.exists():
            print(f"❌ Model not found at: {self.model_path}")
            return False

        try:
            print(f"🔍 Verifying model: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

            if isinstance(checkpoint, dict):
                print(f"✅ Model loaded successfully!")
                print(f"   Keys: {list(checkpoint.keys())}")
                if 'state_dict' in checkpoint:
                    print(f"   Parameters: {len(checkpoint['state_dict'])} layers")
            else:
                print(f"✅ Model loaded (direct state dict)")

            return True

        except Exception as e:
            print(f"❌ Model verification failed: {e}")
            return False


def main():
    """Main download script"""
    print("=" * 60)
    print("🚀 LaMa Model Auto-Downloader")
    print("=" * 60)

    downloader = ModelDownloader(checkpoint_dir="checkpoints")

    # Check if model already exists
    if downloader.model_path.exists():
        print(f"📦 Model already exists at: {downloader.model_path}")

        # Verify it's valid
        if downloader.verify_model():
            print("\n✅ Model is ready to use!")
            return
        else:
            print("\n⚠️  Existing model is corrupted, re-downloading...")
            os.remove(downloader.model_path)

    # Download model
    print("\n📥 Starting download...\n")
    model_path = downloader.download_official_lama()

    if model_path:
        # Verify
        if downloader.verify_model():
            print("\n" + "=" * 60)
            print("✅ SUCCESS! Model downloaded and verified")
            print("=" * 60)
            print(f"📍 Location: {downloader.model_path}")
            print("\n🎯 Next steps:")
            print("   python app.py")
        else:
            print("\n❌ Download completed but verification failed")
    else:
        print("\n❌ Download failed. Please download manually.")


if __name__ == "__main__":
    main()
