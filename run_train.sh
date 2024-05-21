# for ddp
export nproc_per_node=1
export nnodes=1
export node_rank=0

# dataset
export data_path=path_to_name_mapping_file

# model
export model=cnn

if [[ $model =~ maprotonet ]]
  then
    bash ./scripts/train_maprotonet.sh
elif [[ $model =~ mprotonet ]]
  then
    bash ./scripts/train_mprotonet.sh
elif [[ $model =~ xprotonet ]]
  then
    bash ./scripts/train_xprotonet.sh
elif [[ $model =~ protopnet ]]
  then
    bash ./scripts/train_protopnet.sh
elif [[ $model =~ cnn ]]
  then
    bash ./scripts/train_cnn.sh
fi
