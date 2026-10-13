# CIFAR100/20 — Tunable MagMax Label và MagMax, một seed

Config đã chốt: CLIP ViT-B/16 OpenAI; pattern A; seed 5; sequential fine-tuning
20 task × 5 lớp; 2.500 ảnh train/task; 10 epoch/task; batch 128; FP32;
AdamW lr=1e-5, wd=0.1; warmup=500 reset mỗi task; loss CE (ls=0); clip gradient
norm=1.0. Giữ nguyên code training và merging của upstream.

Ở CIFAR100/20 mỗi task chỉ có 200 optimizer steps (20 batch × 10 epoch).
Vì warmup dài 500 steps, task này nằm hoàn toàn trong warmup theo code gốc;
launcher không tự sửa scheduler hoặc rút số epoch.

Chỉ chạy Tunable MagMax (Label) và baseline MagMax, lambda=0.5, cùng 25 target
variants từ 5 phân phối ở mục 4.3/Bảng 2. Lấy 500 mẫu train/task cho similarity;
không giảm dữ liệu fine-tuning xuống 500 ảnh. Mỗi target có tổng 500 ảnh từ test,
10% meta và 90% evaluation; số meta được làm tròn xuống riêng cho từng task.
Một seed nên không có mean/std qua ba lần train như paper.

`configs/target_data_config_main25.json` là config gốc bỏ nhóm target 26.
Giữ nguyên seeds, IDs và ratios của tác giả, gồm `[0.33,0.33,0.33]` và cách phân
bổ phần dư trong code. Thứ tự nhóm 4/5 của JSON upstream ngược Bảng 2; helper
tổng hợp sắp theo tỷ lệ của Bảng 2, không thay đổi quá trình lấy mẫu.

Không tạo Python training mới. Chỉ thêm launcher và helper tổng hợp JSON/CSV.
Một thay đổi logging trong `merge_for_targetdata.py`: dùng WANDB_MODE thay cho
ép online; mặc định ngoài launcher vẫn online như trước. Launcher đặt disabled,
không cần tài khoản WandB. Tương thích checkpoint full-model với torch >=2.6
bằng TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1. Môi trường ai_env khác export tác giả
(torch2.4/CUDA11.8); runtime và pip freeze được lưu cùng kết quả.

## Terminal SSH MICA

MagMax đang dùng GPU0; launcher này mặc định GPU vật lý 1, ánh xạ thành cuda:0
trong process. CUDA_VISIBLE_DEVICES chỉ cho process thấy một GPU.

```bash
cd ~/DrugDrug/tunable-MagMax
conda activate ai_env
git pull --ff-only
export TUNABLE_GPU_ID=1
export MAGMAX_DATA_DIR="$HOME/DrugDrug/MagMax/datasets"
bash scripts/run_tunable_mica.sh check
bash scripts/run_tunable_mica.sh start
bash scripts/run_tunable_mica.sh follow
```

Không cần pip install lại nếu setup check trước đó đã thành công. Dataset và
pretrained cache dùng chung với MagMax. `start` chạy nền ngay cả khi chưa có
dataset: chuẩn bị dataset xong thì tự chuyển sang train, merge/evaluate và tổng
hợp kết quả. Không cần ngồi đợi tải xong để nhập lệnh train.

Nếu job MagMax đang chạy và dataset dùng chung chưa hoàn chỉnh, launcher chờ
job đó tải/giải nén, không ghi đè archive đang được sử dụng. Nếu không có job đó,
launcher tự tải từ nguồn CIFAR chính thức. Dùng curl để thử resume archive tải
dở; nếu server không hỗ trợ resume hoặc checksum sai thì tải bản mới. Kiểm tra
MD5 trước giải nén và kiểm tra torchvision trước training. Dataset đã có đủ và
đúng checksum sẽ được tái sử dụng. Khóa dataset ngăn các launcher này tải trùng.

Nếu đang chạy lệnh Python/curl download thủ công ở terminal khác, dừng lệnh đó
trước `start`; quá trình thủ công không dùng khóa của launcher.
Download dùng CPU, không train cùng lúc với một dataset chưa hoàn chỉnh.
Không dùng chung task checkpoints hoặc zeroshot: Tunable train riêng dưới repo này.

Cache mặc định là `../MagMax/checkpoints/ViT-B-16/cachedir/open_clip`.
Nếu MagMax tải cache ở đường dẫn khác, export TUNABLE_OPENCLIP_CACHE trước start.
Model loader kiểm tra/tải weights bằng OpenCLIP như gốc.

Sau dòng `[STARTED]`, job chạy bằng nohup + flock. Có thể đóng SSH.
Ctrl+C khi `follow` chỉ đóng tail, không dừng job. Hai phương pháp merge/evaluate
chạy lần lượt sau training, không khởi động thêm một lượt training.

```bash
bash scripts/run_tunable_mica.sh status
```

Check hiển thị RAM và disk vì upstream nạp 20 task vectors lên CPU khi merge.
Nên có khoảng 24–32 GiB RAM khả dụng và khoảng 20 GiB disk trống cho checkpoint,
weights/cache/log; đây là dự trù tài nguyên, không phải đo từ run của bạn.
GPU import/probe thành công chưa xác nhận toàn bộ training chạy thành công.

## Kết quả và xử lý gián đoạn

Output: `outs/tunable_CIFAR100_20_A_seed5/`.

- `summary.csv`, `summary.json`: overall top-1 (%) trung bình 5 variants mỗi
  phân phối, rồi trung bình 5 phân phối. Không lấy macro accuracy theo task làm
  metric chính. Ghi rõ một seed, không báo std qua seeds.
- `results/`: 50 JSON gốc (25 Label + 25 MagMax), chứa accuracy theo task,
  macro/overall accuracy, preference và thông tin target sampling.
- `download.log`, `train.log`, `label.log`, `magmax.log`, `nohup.log`: log từng giai đoạn.
- `stage.txt`: dataset → train → label → magmax → summary → done.
- `runtime.json`, `environment.txt`, `source_commit.txt`, `commands.txt`,
  `target_config.json`, `gpu.txt`, `data_dir.txt`: thông tin tái lập.
- `exit_code.txt`: 0 nghĩa là toàn bộ pipeline và tổng hợp kết quả thành công.

`start` cho phép thử lại nếu chỉ bước dataset lỗi và chưa có task checkpoint;
giữ lại exit code cũ và append log. Các lần chạy đã bắt đầu training vẫn được
bảo vệ như trước. Nếu đủ 20 task checkpoints
và merge/evaluation bị gián đoạn, chạy lại chỉ phần merge:

```bash
bash scripts/run_tunable_mica.sh merge
bash scripts/run_tunable_mica.sh follow
```

Code gốc bỏ qua JSON đã có; helper chỉ xuất summary khi đủ cả 50 JSON và kiểm
tra hai phương pháp có cùng thông tin lấy mẫu target. Nếu 1–19 checkpoints,
launcher dừng vì upstream không lưu RNG/optimizer để resume chính xác.
Giữ lại thư mục checkpoint và log để xem nguyên nhân trước khi chọn chạy lại.
Launcher không tự xóa hoặc tự resume một chuỗi train dang dở.

Nguồn: https://arxiv.org/html/2605.20803v3 và
https://github.com/KeiHiroshima/tunable-magmax
