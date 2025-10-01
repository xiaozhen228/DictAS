(
nohup python train.py --dataset visa --train_data_path ./dataset/mvisa/data --anomaly_source_path ./datasets/DTD/images \
--save_path ./exps/train_visa/222/vit_large_14_336 --gen_anomaly_rate 0.7 --lambda1 0.1 --lambda2 0.1 \
--config_path ./open_clip_local/model_configs/ViT-L-14-336.json --model ViT-L-14-336 \
--features_list 6 12 18 24 --pretrained openai --image_size 336  --batch_size 24 --print_freq 1 --seed 222 \
--epoch 30 --save_freq 1 --learning_rate 0.0001 --device_id 0 --pretrained_path ./pretrained_weight/ViT-L-14-336px.pt
) > ./log_train_visa.out 2>&1 &