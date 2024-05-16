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
