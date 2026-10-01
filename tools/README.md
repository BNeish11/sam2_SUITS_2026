## SAM 2 toolkits

This directory provides toolkits for additional SAM 2 use cases.

### Semi-supervised VOS inference

The `vos_inference.py` script can be used to generate predictions for semi-supervised video object segmentation (VOS) evaluation on datasets such as [DAVIS](https://davischallenge.org/index.html), [MOSE](https://henghuiding.github.io/MOSE/) or the SA-V dataset.

After installing SAM 2 and its dependencies, it can be used as follows ([DAVIS 2017 dataset](https://davischallenge.org/davis2017/code.html) as an example). This script saves the prediction PNG files to the `--output_mask_dir`.
```bash
python ./tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
  --base_video_dir /path-to-davis-2017/JPEGImages/480p \
  --input_mask_dir /path-to-davis-2017/Annotations/480p \
  --video_list_file /path-to-davis-2017/ImageSets/2017/val.txt \
  --output_mask_dir ./outputs/davis_2017_pred_pngs
```
(replace `/path-to-davis-2017` with the path to DAVIS 2017 dataset)

To evaluate on the SA-V dataset with per-object PNG files for the object masks, we need to **add the `--per_obj_png_file` flag** as follows (using SA-V val as an example). This script will also save per-object PNG files for the output masks under the `--per_obj_png_file` flag.
```bash
python ./tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
  --base_video_dir /path-to-sav-val/JPEGImages_24fps \
  --input_mask_dir /path-to-sav-val/Annotations_6fps \
  --video_list_file /path-to-sav-val/sav_val.txt \
  --per_obj_png_file \
  --output_mask_dir ./outputs/sav_val_pred_pngs
```
(replace `/path-to-sav-val` with the path to SA-V val)

Then, we can use the evaluation tools or servers for each dataset to get the performance of the prediction PNG files above.

Note: by default, the `vos_inference.py` script above assumes that all objects to track already appear on frame 0 in each video (as is the case in DAVIS, MOSE or SA-V). **For VOS datasets that don't have all objects to track appearing in the first frame (such as LVOS or YouTube-VOS), please add the `--track_object_appearing_later_in_video` flag when using `vos_inference.py`**.

### Rover workflow

This workflow restores the original order: train YOLO on the labeled training folder, run YOLO on the rover pictures, then let SAM2 segment using the YOLO box prompt.

1) Train YOLO on the labeled rover dataset

```bash
python tools/train_rover_yolo.py \
  --dataset-yaml "training/rover images/rover_retrain_dataset/dataset.yaml" \
  --base-model yolo26n.pt \
  --project runs/yolo_runs \
  --run-name rover_det_train \
  --epochs 80 \
  --imgsz 960 \
  --batch 8 \
  --seed 42
```

2) Run YOLO + SAM2 on the 7 rover pictures

```bash
python tools/run_rover_image_batch.py \
  --images "rover pictures" \
  --yolo-model runs/yolo_runs/rover_det_train/weights/best.pt \
  --points-file picked_rover_image_points.txt \
  --out runs/rover_batch \
  --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --conf 0.2 \
  --iou 0.45 \
  --imgsz 960
```

`run_rover_image_batch.py` uses manual prompts first (if `--points-file` is provided), otherwise it uses the YOLO rover box as the SAM2 prompt.

Manual prompt format (positive and negative points per image):

```text
# IMAGE Rover_1_parking_lot.jpeg
# POS
(1455, 1485)
(1440, 1520)
# NEG
(1200, 1400)

# IMAGE rover_5_tree2.jpeg
# POS
(2018, 1560)
```

- `# POS` points are foreground rover clicks.
- `# NEG` points are background clicks to suppress over-segmentation.
- If `# POS` / `# NEG` headers are omitted, points default to positive (backward compatible).

To click points interactively, run:

```bash
python tools/pick_rover_image_points.py
```
