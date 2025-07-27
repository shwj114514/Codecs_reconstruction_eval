export http_proxy='http://10.2.254.3:8888'
export https_proxy='http://10.2.254.3:8888'

export CUDA_VISIBLE_DEVICES="4"

DEVICE="cuda:0"

RECON_OUTPUT_FOLDER="recon_rebuttal_all"
EXP_FOLDER="exp_recon"

run_model() {
    local MODEL=$1
    local N_QUANTIZERS=$2

    AUDIO_FOLDER="${EXP_FOLDER}/test-clean"


    echo "Cal entropy model: $MODEL with $N_QUANTIZERS quantizers"

    python metrics/compute_entropy.py \
        --device $DEVICE \
        --folder "$AUDIO_FOLDER" \
        --model $MODEL \
        --n_quantizers $N_QUANTIZERS
}

# 24k
run_model "WavTokenizer600" 0
run_model "WavTokenizer320" 0

run_model "Mimi" 8
run_model "Mimi" 5

run_model "DAC_16k" 2


run_model "SpeachTokenzier" 1
run_model "SpeachTokenzier" 2

run_model "Encodec" 2

# 16k
run_model "StableCodec" 0

# run_model "Xcodec2" 0

run_model "BigCodec" 0
