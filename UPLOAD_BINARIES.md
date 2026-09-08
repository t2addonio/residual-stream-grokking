# Binary files this connector cannot push

GitHub's connected file API here is text-only. The v6 PDF, figures, and result archives must be added from a browser or a local `git push`.

## Fastest: GitHub web UI (30 seconds)

1. Open https://github.com/t2addonio/residual-stream-grokking/upload/main
2. Create folders by naming files:
   - `papers/Residual_Stream_Component_Grokking_v6_full.pdf`
   - `papers/figures/fig5_capacity_scaling.png` (same for fig6–fig9)
   - `results/archives/<archive>.tar.gz`
3. Drop the files from the download zip in this chat, commit.

## Local git

```bash
git clone https://github.com/t2addonio/residual-stream-grokking.git
cd residual-stream-grokking
mkdir -p papers/figures results/archives
# unzip github_upload_binaries.zip into those folders
git add papers results/archives
git commit -m "Add v6 PDF, figures, and result archives"
git push
```
