# training
seed=0
epoch=100
batch_size=32
lr=0.001
wd=0.01

# model
prototype_shape='(30, 128, 1, 1, 1)'
coefs="[{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05}]"
backbone=resnet152
n_res_block=1

python -m torch.distributed.launch --nproc_per_node $nproc_per_node --nnodes $nnodes --node_rank $node_rank ./tumor_cls.py \
  -d $data_path \
  -m $model \
  -n $epoch \
  -b $batch_size \
  --lr $lr \
  --wd $wd \
  --backbone $backbone \
  --n_res_block $n_res_block \
  --prototype_shape "$prototype_shape" \
  --f_dist cos \
  --coefs "$coefs" \
  --save-model 1 \
  -s $seed