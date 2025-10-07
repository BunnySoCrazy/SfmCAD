python train.py -e shape_table_loft --grid 16  --epoch 20 -g 5
python train.py -e shape_table_loft --continue --grid 32 --ckpt best_16  --epoch 20  -g 5
python train.py -e shape_table_loft --continue --grid 64 --ckpt best_32  --epoch 30 -g 5

python fine-tuning.py -e shape_table_loft -g 5 --start 0 --end 2 --epoches 10 --grid 16 -c best_64  --test_data 
python fine-tuning.py -e shape_table_loft -g 5 --start 0 --end 2 --epoches 50 --grid 32 -c best_16 -l  --test_data 
python fine-tuning.py -e shape_table_loft -g 5 --start 0 --end 2 --epoches 50 --grid 64 -c best_32 -l  --test_data 

python test.py -e shape_table_loft -g 5 -c  best_64 --mc_threshold 0.55 --start 0 --end 2 --grid_sample 128    
