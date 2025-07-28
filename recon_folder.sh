DEVICE="cuda:7"

RECON_OUTPUT_FOLDER="recon_rebuttal_all"
EXP_FOLDER="exp_recon"

run_model() {
    local MODEL=$1
    local N_QUANTIZERS=$2

    AUDIO_FOLDER="${EXP_FOLDER}/test-clean"

    echo "Processing model: $MODEL with $N_QUANTIZERS quantizers"

    python recon_folder.py \
        --device $DEVICE \
        --input_folder "$AUDIO_FOLDER" \
        --model $MODEL \
        --output_folder "${EXP_FOLDER}/${MODEL}_${N_QUANTIZERS}" \
        --n_quantizers $N_QUANTIZERS
}

# 24k
# run_model "WavTokenizer600" 0
# run_model "Mimi" 8
# run_model "Mimi" 6

run_model "DAC_24k" 9

# run_model "SpeachTokenzier" 1
# run_model "SpeachTokenzier" 2

# run_model "Encodec" 2
# run_model "Encodec" 8


# 16k
# run_model "StableCodec" 0

# run_model "Xcodec2" 0

# run_model "BigCodec" 0


