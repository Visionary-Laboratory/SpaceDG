CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun ../run.py \
  --model InternVL3_5-8B \
  --data spacedg_bench \
  --mode all \
  --work-dir ../outputs_spacedg \
  --reuse