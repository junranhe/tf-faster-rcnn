#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB=$3
TEST_IMDB=$4
STEPSIZE=80000
ITERS=200000
ANCHORS="[8,16,32]"
RATIOS="[0.5,1,2]"

LOG="experiments/logs/${NET}_${TRAIN_IMDB##*/}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    NET_FINAL=output/${NET}/${TRAIN_IMDB##*/}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
    NET_FINAL=output/${NET}/${TRAIN_IMDB##*/}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
    if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID}  python ./tools/trainval_net.py \
            --weight data/imagenet_weights/${NET}.ckpt \
            --imdb ${TRAIN_IMDB##*/} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg experiments/cfgs/${NET}.yml \
            --tag ${EXTRA_ARGS_SLUG} \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID}  python ./tools/trainval_net.py \
            --weight data/imagenet_weights/${NET}.ckpt \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg experiments/cfgs/${NET}.yml \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    fi
fi

./experiments/scripts/test_faster_rcnn.sh $@
