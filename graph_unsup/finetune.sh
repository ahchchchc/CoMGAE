dataset=$"IMDB-BINARY" #


for runseed in 30 31 32 33 34
do
python CoMGAE_joint_pretrain.py --seed $runseed --dataset $dataset\
&& python CoMGAE_finetune_SVM.py  --seed $runseed --dataset $dataset
done

# --lr 1e-3 --epochs 100