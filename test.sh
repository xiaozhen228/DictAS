(
nohup python test.py --dataset mvtec --data_path ./dataset/mvisa/data --anomaly_source_path ./datasets/DTD/images \
--save_path ./results/test_mvtec/222/vit_large_14_336  \
--config_path ./open_clip_local/model_configs/ViT-L-14-336.json --model ViT-L-14-336 \
--features_list 6 12 18 24 --pretrained openai --image_size 336  --seed 222 \
--checkpoint_path ./DictAS_weight/train_visa.pth --k_shot 4 \
--device_id 0 --pretrained_path ./pretrained_weight/ViT-L-14-336px.pt
) > ./log_test_mvtec.out 2>&1 &