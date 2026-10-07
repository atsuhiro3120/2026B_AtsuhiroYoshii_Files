import os
import glob
import time
import numpy as np
import torch
import scipy.ndimage as ndimage  # 境界線を太くするために追加
from PIL import Image
from tqdm import tqdm  # 進捗バー用ライブラリ
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# 1. 環境設定とデバイスの選択
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# 2. SAM 2 モデルの構築と初期化
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

print("SAM 2 モデルを読み込み中...")
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.85,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=25.0,
    use_m2m=True,
)

# 3. フォルダの設定と作成
input_dir = "input_artifact"
output_dir = "output_artifact_2px_edge"
os.makedirs(output_dir, exist_ok=True)

# 入力画像の一覧取得
extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
image_paths.sort()

total_images = len(image_paths)
print(f"処理対象の画像: {total_images}枚\n")

# 4. ループ処理（進捗バー付き）
start_all_time = time.time()

# tqdmを使ってコンソールに進捗バーを表示
for img_path in tqdm(image_paths, desc="全体の進捗", unit="img"):
    filename = os.path.basename(img_path)
    basename = os.path.splitext(filename)[0]
    
    # 画像の読み込み
    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))
    h, w, _ = image.shape

    # マスクの自動生成
    masks = mask_generator.generate(image)

    # 所属マップの作成 (-1で初期化)
    assignment_map = np.full((h, w), -1, dtype=int)
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    # 小さいパーツで上書きしていく
    for i, ann in enumerate(sorted_masks):
        assignment_map[ann['segmentation']] = i

    # カラー更新用画像の作成
    updated_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 各領域の平均色を計算してピクセルを更新
    for i, ann in enumerate(sorted_masks):
        target_pixels_mask = (assignment_map == i)
        if np.any(target_pixels_mask):
            mean_color = image[target_pixels_mask].mean(axis=0).astype(np.uint8)
            updated_img[target_pixels_mask] = mean_color

    # 未検出の空白領域もその平均色で更新
    unmapped_mask = (assignment_map == -1)
    if np.any(unmapped_mask):
        unmapped_mean_color = image[unmapped_mask].mean(axis=0).astype(np.uint8)
        updated_img[unmapped_mask] = unmapped_mean_color

    # 【修正】境界線を抽出し、任意pxの太さに拡張する処理
    edge_mask = np.zeros((h, w), dtype=bool)
    # 1px分の境界線を抽出
    edge_mask[:, :-1] |= (assignment_map[:, :-1] != assignment_map[:, 1:])
    edge_mask[:-1, :] |= (assignment_map[:-1, :] != assignment_map[1:, :])
    
    #線の太さ
    structure = np.ones((2, 2), dtype=bool)
    edge_mask_3px = ndimage.binary_dilation(edge_mask, structure=structure)
    
    # 境界線を黒[0, 0, 0]で一括上書き
    updated_img[edge_mask_3px] = [0, 0, 0]

    # 保存
    output_path = os.path.join(output_dir, f"{basename}.png")
    output_image = Image.fromarray(updated_img)
    output_image.save(output_path)

end_all_time = time.time()
elapsed_total = end_all_time - start_all_time

print("\n" + "="*40)
print(" すべての画像処理が完了しました！")
print(f" 総処理時間: {elapsed_total:.2f} 秒")
print(f" 平均処理速度: {elapsed_total / total_images:.2f} 秒/枚")
print(f" 出力先フォルダ: '{output_dir}'")
print("="*40)