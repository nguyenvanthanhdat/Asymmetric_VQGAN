python main.py --base configs/autoencoder/autoencoder_kl_32x32x4.yaml \
    -t \
    --gpus 0,1 \
    --epoch 1 \
    --tag Base_train_from_source