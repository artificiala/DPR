for ENCODER in hf_thwiki_20210520_bert hf_thwiki_20210520_news_bert
do 
    for CHUNK_SIZE in 150 300
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
                HN_TEXT=""
                export EXP_NAME="exp002.v2.batched.${ENCODER}.thwiki-split_w${CHUNK_SIZE}"
                train_dataset="thwiki_15k_split_w${CHUNK_SIZE}_train"
                validation_dataset="thwiki_15k_split_w${CHUNK_SIZE}_validation"
            else
                HN_TEXT="hn_bm25_${HN}_"
                export EXP_NAME="exp002.v2.batched.${ENCODER}.thwiki-split_w${CHUNK_SIZE}_hn-${HN}"
                train_dataset="thwiki_15k_split_w${CHUNK_SIZE}_${HN_TEXT}train"
                validation_dataset="thwiki_15k_split_w${CHUNK_SIZE}_${HN_TEXT}validation"

            fi
            mkdir -p /workspace/logs/dpr/${EXP_NAME}
            mkdir -p /workspace/checkpoints/dpr/${EXP_NAME}
            cd /workspace/DPR
            rm  /workspace/logs/dpr/${EXP_NAME}/encoder_training.log

            echo "EXP NAME: ${EXP_NAME}"
            

    
            

            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python generate_dense_embeddings.py \
                model_file=/workspace/checkpoints/dpr/${EXP_NAME}/_dpr_biencoder.best \
                ctx_src=dpr_thwiki-20171217_split-${CHUNK_SIZE} \
                shard_id=0 num_shards=1 \
                batch_size=680 \
                out_file=/workspace/embed/dpr_wiki/embed.batched.${EXP_NAME}_best@ |& tee -a /workspace/logs/generate_dense_embeddings/${RESULT_NAME}


            CUDA_VISIBLE_DEVICE=1 python dense_retriever.py \
                model_file=/workspace/checkpoints/dpr/${EXP_NAME}/_dpr_biencoder.best \
                qa_dataset=thwiki_qas_15k_train \
                ctx_datatsets=[dpr_thwiki-20210520_split-300] \
                encoded_ctx_files=["/workspace/embed/dpr_wiki/embed.batched.${EXP_NAME}_best@_0"] \
                out_file=/workspace/outputs/dpr_retriever_evaluation/${EXP_NAME}/result.train
            
            CUDA_VISIBLE_DEVICE=1 python dense_retriever.py \
                model_file=/workspace/checkpoints/dpr/${EXP_NAME}/_dpr_biencoder.best \
                qa_dataset=thwiki_qas_15k_validation \
                ctx_datatsets=[dpr_thwiki-20210520_split-300] \
                encoded_ctx_files=["/workspace/embed/dpr_wiki/embed.batched.${EXP_NAME}_best@_0"] \
                out_file=/workspace/outputs/dpr_retriever_evaluation/${EXP_NAME}/result.validation
            
            CUDA_VISIBLE_DEVICE=1 python dense_retriever.py \
                model_file=/workspace/checkpoints/dpr/${EXP_NAME}/_dpr_biencoder.best \
                qa_dataset=thwiki_qas_15k_test \
                ctx_datatsets=[dpr_thwiki-20210520_split-300] \
                encoded_ctx_files=["/workspace/embed/dpr_wiki/embed.batched.${EXP_NAME}_best@_0"] \
                out_file=/workspace/outputs/dpr_retriever_evaluation/${EXP_NAME}/result.test

        done
    done
done