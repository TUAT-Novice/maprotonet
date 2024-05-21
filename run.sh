# for ddp
export nproc_per_node=1
export nnodes=1
export node_rank=0

# dataset
export data_path=D:/data/BraTS2018/MICCAI_BraTS_2018_Data_Training

# model
export model=maprotonet

# provide the model hash code only when you want to evaluate, such as maprotonet_eefb07f7
# export load_model=hash_code_of_the_model

if [[ $model =~ maprotonet ]]
  then
    bash ./scripts/run_maprotonet.sh
elif [[ $model =~ mprotonet ]]
  then
  bash ./scripts/run_mprotonet.sh
elif [[ $model =~ xprotonet ]]
  then
    bash ./scripts/run_xprotonet.sh
elif [[ $model =~ protopnet ]]
  then
    bash ./scripts/run_protopnet.sh
elif [[ $model =~ cnn ]]
  then
    bash ./scripts/run_cnn.sh
fi
