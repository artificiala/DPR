for ENCODER in hf_thwiki_20210520_bert hf_thwiki_20210520_news_bert
do 
    for CHUNK_SIZE in 300 150
    do
        for HN in 0 1 2 10
        do

            export HYDRA_FULL_ERROR=1
           
            export train_config=""
            export train_dataset=""
            export validation_dataset=""
            export HN_TEXT=""
            if [ $HN == 0 ]
            then
                train_config="biencoder_inbatch.thwiki-15k.v1"
                HN_TEXT=""
                export EXP_NAME="exp002.v1.batched_thwiki-20210520-bert_thwiki-split_w${CHUNK_SIZE}"
                train_dataset="thwiki_15k_split_w${CHUNK_SIZE}_train"
                validation_dataset="thwiki_15k_split_w${CHUNK_SIZE}_validation"
            else
                train_config="biencoder_inbatch.thwiki-15k_hn-${HN}.v1"
                HN_TEXT="hn_bm25_${HN}_"cd
                export EXP_NAME="exp002.v1.batched_thwiki-20210520-bert_thwiki-split_w${CHUNK_SIZE}_hn-${HN}"
                train_dataset="thwiki_15k_split_w${CHUNK_SIZE}_${HN_TEXT}train"
                validation_dataset="thwiki_15k_split_w${CHUNK_SIZE}_${HN_TEXT}validation"

            fi
            mkdir -p /workspace/logs/dpr/${EXP_NAME}
            mkdir -p /workspace/checkpoints/dpr/${EXP_NAME}
            cd /workspace/DPR
            rm  /workspace/logs/dpr/${EXP_NAME}/encoder_training.log


            CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 train_dense_encoder.py \
            train=${train_config} \
            encoder=${ENCODER} \
            train_datasets=[${train_dataset}] \
            dev_datasets=[${validation_dataset}] \
            output_dir=/workspace/checkpoints/dpr/${EXP_NAME} |& tee -a /workspace/logs/dpr/${EXP_NAME}/encoder_training.log

            mv /workspace/checkpoints/dpr/${EXP_NAME}/dpr_biencoder.best /workspace/checkpoints/dpr/${EXP_NAME}/_dpr_biencoder.best
            rm /workspace/checkpoints/dpr/${EXP_NAME}/dpr_biencoder.*
        done
    done
done