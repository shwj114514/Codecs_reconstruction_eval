export http_proxy='http://10.2.254.3:8888'
export https_proxy='http://10.2.254.3:8888'

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


EXP_FOLDER="exp_recon"

# export TRANSFORMERS_CACHE=""


run_model() {
    local MODEL=$1
    local N_QUANTIZERS=$2

    AUDIO_FOLDER="${EXP_FOLDER}/test-clean"

    echo "Processing model: $MODEL with $N_QUANTIZERS quantizers"

    torchrun --nproc_per_node=${num_gpus}  recon_folder_multi.py \
        --input_folder "$AUDIO_FOLDER" \
        --model $MODEL \
        --output_folder "${EXP_FOLDER}/${MODEL}_${N_QUANTIZERS}" \
        --n_quantizers $N_QUANTIZERS
}

# 24k
# run_model "WavTokenizer600" 0
# run_model "WavTokenizer320" 0

# run_model "Mimi" 8
# run_model "Mimi" 6

# run_model "DAC_16k" 2
# run_model "DAC_24k" 1

run_model "DAC_24k" 9

# run_model "SpeachTokenzier" 1
# run_model "SpeachTokenzier" 2

# run_model "Encodec" 8 &
# run_model "DAC_24k" 8 &


# 16k
# run_model "StableCodec" 0
# run_model "StableCodec_base" 0

# run_model "StableCodec_base_400bps" 0
# run_model "StableCodec_base_700bps" 0
# run_model "StableCodec_base_1000bps" 0

# run_model "Xcodec" 1
# run_model "Xcodec" 2
# run_model "SemanticCodec_700bps" 0

# run_model "Xcodec2" 0

# run_model "BigCodec" 0


