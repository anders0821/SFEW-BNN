PATH=/usr/local/cuda-7.5/bin:$PATH
LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH 

mkdir -p ./train_val/

for N_HIDDEN_LAYERS in {2..2}
do
for NUM_UNITS in 1500
do
for OUTPUT_TYPE in C
do
for MAIN_LOSS_TYPE in H
do
for LAMBDA in 0
do
for FOLD in {1..1}
do

################################################################################
fn="./train_val/$N_HIDDEN_LAYERS-$NUM_UNITS-$OUTPUT_TYPE-$MAIN_LOSS_TYPE-$LAMBDA-$FOLD.txt"

# 尝试抢占任务
python ./touch_fail_if_exist.py $fn

# 抢占成功
if [ $? == 0 ]
then
  # 启动任务
  echo $fn
  STARTTIME=$(date +%s)
  PATH=/usr/local/cuda-7.5/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH THEANO_FLAGS=floatX=float32,device=gpu python -u ./train_val_trial.py $N_HIDDEN_LAYERS $NUM_UNITS $OUTPUT_TYPE $MAIN_LOSS_TYPE $LAMBDA $FOLD |& tee $fn

  # trial出错则退出，一般是内存显存不足错误，导致无法并行
  if [ $? != 0 ]
  then
    cat $fn
    echo TRIAL ERROR, DELETE TRIAL LOG, EXIT
    rm $fn
    exit
  fi

  # 结束任务 汇报时间
  cat $fn | tail -n 10
  ENDTIME=$(date +%s)
  echo $(($ENDTIME - $STARTTIME)) sec
fi
################################################################################

done
done
done
done
done
done

