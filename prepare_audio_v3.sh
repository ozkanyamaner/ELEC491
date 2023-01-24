#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

train_split=train
valid_split=valid
test_split=test

#all_splits=($train_split)

#if [[ -f "$source_dir/valid.tsv" ]]; then
#    all_splits+=('valid')
#fi

#if [[ -f "$source_dir/test.tsv" ]]; then
#    all_splits+=('test')
#fi

#echo "processing splits: $all_splits"

setopt shwordsplit

#for split in $all_splits; do
# python examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py $source_dir --split $split \
#  --save-dir $tgt_dir --checkpoint $model --layer $layer
#done

python examples/wav2vec/unsupervised/scripts/wav2vec_from_mp4.py "/home/ozkan/Desktop/ELEC491/recording_test" --split $test_split \
  --save-dir "/home/ozkan/Desktop/ELEC491/out" --checkpoint "/home/ozkan/Desktop/ELEC491/fairseq/wav2vec_vox_new.pt" --layer 14

#mkdir -p $tgt_dir/w2v2

# Consider spliting corpus into chuncks for large corpus, see HuBERT preprocessing for more details
#python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_w2v2_feature.py $tgt_dir $train_split $model $layer 1 0 $tgt_dir/w2v2
#python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py $tgt_dir/w2v2 $train_split 1 $tgt_dir/w2v2/cls$dim $dim --percent -1

#python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py \
#  $tgt_dir/w2v2 $train_split $tgt_dir/w2v2/cls$dim 1 0 $tgt_dir/w2v2/cls${dim}_idx
#cp $tgt_dir/w2v2/cls${dim}_idx/${train_split}_0_1.km $tgt_dir/$train_split.km
