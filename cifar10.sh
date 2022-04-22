for i in `seq 10`  
do  
    python main.py --dataset cifar10 --normal_class `expr $i - 1` --regular 2
done 
