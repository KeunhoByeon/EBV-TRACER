# EBV-TRACER

## Environment Setup
```bash
conda create -n ebv_tracer python=3.9
conda activate ebv_tracer
pip install -r requirements.txt
```

## Patch Generation
```bash
python make_patches.py \
--input_svs_dir INPUT_SVS_DIR \
--save_patch_dir SAVE_PATCH_DIR \
--n_jobs 16
```

## Two stage stain normalization
```bash
python two_stage_stain_normalize.py \
--target_svs_dir TARGET_SLIDE_DIR \
--input_svs_dir SOURCE_SLIDE_DIR \
--input_patch_dir SOURCE_PATCH_DIR \
--output_patch_dir OUTPUT_PATCH_DIR \
--output_thumbnail_dir OUTPUT_THUMBNAIL_DIR \
--n_jobs 16
```

## EBV-GC Classification
Download pre-trained EfficientNet B1 model weights from [this link](https://github.com/KeunhoByeon/EBV-TRACER/releases/download/v1/net_5000.pth) and save ot in the "./checkpoints" directory.

#### Run Inference
```bash
python inference.py \
--workers 16 \
--batch_size 128 \
--checkpoint ./checkpoints/net_5000.pth \
--results ./results.csv \
--data PATCH_DIR
```
