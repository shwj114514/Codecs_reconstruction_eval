export http_proxy='http://10.2.254.3:8888'
export https_proxy='http://10.2.254.3:8888'

export CUDA_VISIBLE_DEVICES="3"

DEVICE="cuda:0"

RECON_OUTPUT_FOLDER="recon_rebuttal_all"
EXP_FOLDER="exp_recon"

run_model() {
    local MODEL=$1
    local N_QUANTIZERS=$2

    AUDIO_FOLDER="${EXP_FOLDER}/test-clean"
    # AUDIO_FOLDER="${EXP_FOLDER}/tmp"


    echo "Cal entropy model: $MODEL with $N_QUANTIZERS quantizers"

    python metrics/compute_usage.py \
        --device $DEVICE \
        --folder "$AUDIO_FOLDER" \
        --model $MODEL \
        --n_quantizers $N_QUANTIZERS
}

# run_model "Xcodec2" 0
# run_model "Mimi" 6
# run_model "SpeachTokenzier" 1
# run_model "SpeachTokenzier" 2
# run_model "DAC_16k" 2
# run_model "DAC_24k" 1
# run_model "Encodec" 1

# run_model "WavTokenizer600" 0
# run_model "WavTokenizer320" 0
# run_model "BigCodec" 0

# run_model "StableCodec" 0
# run_model "StableCodec_base" 0

# run_model "StableCodec_base_400bps" 0
# run_model "StableCodec_base_700bps" 0
# run_model "StableCodec_base_1000bps" 0


# run_model "Xcodec" 1
# run_model "Xcodec" 2
# run_model "SemanticCsodec_700bps" 0


# run_model "Encodec" 8
# run_model "DAC_24k" 8 

# run_model "DAC_16k" 1
# run_model "DAC_16k" 12

# nohup bash run_usage.sh > run_usage.txt 2>&1 &

# run_model "Encodec" 2
# run_model "DAC_16k" 2
run_model "WavTokenizer600" 0