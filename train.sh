CUDA_VISIBLE_DEVICES=1 nohup python -u train.py \
--lr_encoder 2e-6 \
--lr_others 2e-4 \
--n_ctx 12 \
--batch_size  8 \
--output_dir 'results/' \
--data_dir 'data/projection/' \
--img_length_read 6 \
--num_epochs 30 \
--k_fold_num 5 \
--class_token_position 'front' \
> log_MATE3D.txt 2>&1 &