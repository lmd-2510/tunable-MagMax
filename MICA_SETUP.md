# Chuẩn bị Tunable MagMax trên MICA

Source chính thức: https://github.com/KeiHiroshima/tunable-magmax
Commit upstream khi clone: `c8bddf3a07dd6f3b5dfb9ed617665ecefb77c758`.
Repo cá nhân: https://github.com/lmd-2510/tunable-MagMax

Giữ nguyên code training, merging, scripts và configs của tác giả. Chỉ thêm
`requirements.txt`, `scripts/setup_mica.sh` và hướng dẫn này để chuẩn bị `ai_env`.
Setup không chọn config thực nghiệm và không train/download dữ liệu hay pretrained weights.

## Trên terminal SSH MICA

```bash
mkdir -p ~/DrugDrug
cd ~/DrugDrug
git clone https://github.com/lmd-2510/tunable-MagMax.git
cd tunable-MagMax
conda activate ai_env
bash scripts/setup_mica.sh install
```

Nếu repo đã clone, thay bước clone bằng `cd ~/DrugDrug/tunable-MagMax` rồi `git pull --ff-only`.
`install` giữ bản torch/torchvision/torchaudio đã có bằng pip constraints, và cài
các package dùng trong training/merge theo phiên bản từ `environment.yml` upstream.
Các package khác sẽ được thay đổi nếu chưa đúng phiên bản ghi trong requirements.
Các dependency CLIP/NumPy trùng bản đã cài cho MagMax; thêm WandB/GeomLoss/Optuna/Matplotlib.
Đây là cách tái sử dụng `ai_env`; không khẳng định giống nguyên môi trường của tác giả.

Kiểm tra trên một GPU được cấp cho bạn:

```bash
export TUNABLE_GPU_ID=1
bash scripts/setup_mica.sh check
```

GPU chỉ nhận giá trị 0 hoặc 1. Chọn GPU đang được phép sử dụng; GPU0 hiện đã dùng cho
MagMax thì giữ run đó hoạt động. `check` chỉ chạy phép CUDA rất nhỏ, không train.
Lưu thông tin môi trường vào `outs/mica_setup/environment.json`.

Mặc định `MAGMAX_BASE_DIR` là root repo, `MAGMAX_DATA_DIR` là `datasets/` của repo.
Bạn có thể cấu hình đường dẫn bằng environment variables mà không sửa `src/config.py`:

```bash
export MAGMAX_BASE_DIR="$PWD"
export MAGMAX_DATA_DIR="$HOME/DrugDrug/MagMax/datasets"
bash scripts/setup_mica.sh check
```

Đường dẫn data ở trên dùng chung CIFAR-100 đã tải cho MagMax; trước khi chạy thật cần
bảo đảm quá trình tải dữ liệu đó hoàn tất. Không tự dùng chung checkpoint: cần đối chiếu
task sequence, seed, training, định dạng model và class order khi chốt config.

Setup đặt `WANDB_MODE=disabled` trong tiến trình và bật tương thích load checkpoint
full-model gốc với PyTorch >=2.6 qua `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
Script setup kết thúc thì các biến này không tự lưu vào terminal; launcher training
sau khi chốt config sẽ cần export chúng. GPU vật lý được chọn sẽ là `cuda:0` trong
tiến trình sau `CUDA_VISIBLE_DEVICES`; CLI `--gpu_id` của training/merging sẽ dùng 0.

## Sau setup

Chốt dataset, số task, task sequence A/B/C, seed, similarity metric, target configuration,
số target samples và số train samples/task trước khi viết launcher check/start/status/follow.
Các script upstream hiện có mặc định CIFAR100/5, patternA, seed3; chúng chưa phải config
đã chốt cho run của bạn. Không chạy `scripts/finetune_merge.sh` cho đến khi đối chiếu config.

Source gốc vẫn có `environment.yml` nếu cần tái tạo môi trường theo export tác giả.
Chưa thực hiện cài `ai_env`, GPU preflight hoặc training trên MICA từ máy Windows.
