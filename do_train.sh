# TODO: Replace with where you downloaded your resnet_v2_50.
PRETRAINED=models/resnet_v2_50.ckpt
# TODO: Replace with where you generated tf_record!
DATA_DIR=datasets/tf_datasets/

#CMD="python -m src.main --d_lr 1e-4 --e_lr 1e-5 --log_img_step 1000 --pretrained_model_path=${PRETRAINED} --data_dir ${DATA_DIR} --e_loss_weight 60. --batch_size=64 --use_3d_label True --e_3d_weight 60. --datasets lsp,lsp_ext,mpii,h36m,coco,mpi_inf_3dhp --epoch 75 --log_dir logs"
#CMD="python3 src/main.py --d_lr 1e-4 --e_lr 1e-5 --log_img_step 100 --pretrained_model_path=${PRETRAINED} --data_dir ${DATA_DIR} --e_loss_weight 60. --batch_size=64 --use_3d_label True --e_3d_weight 60. --datasets lsp,lsp_ext,mpii,coco,mpi_inf_3dhp --epoch 75 --log_dir logs"

# To pick up training/training from a previous model, set LP
LP='models/pre'
# CMD="python -m src.main --d_lr 1e-4 --e_lr 1e-5 --log_img_step 1000 --load_path=${LP} --e_loss_weight 60. --batch_size=64 --use_3d_label True --e_3d_weight 60. --datasets lsp lsp_ext mpii h36m coco mpi_inf_3dhp --epoch 75"
CMD="python3 src/main.py --d_lr 1e-4 --e_lr 1e-5 --log_img_step 100 --pretrained_model_path=${PRETRAINED} --data_dir ${DATA_DIR} --e_loss_weight 60. --batch_size=64 --use_3d_label True --e_3d_weight 60. --datasets lsp,lsp_ext,mpii,coco,mpi_inf_3dhp,imdb --epoch 75 --log_dir logs"


echo $CMD
$CMD

