export CUDA_VISIBLE_DEVICES="7"


run_evaluation() {
    local ref_dir=$1
    local gen_dir=$2

    echo "Reference Directory: ${ref_dir}"
    echo "Generated Directory: ${gen_dir}"

    echo "Compute loss"
    python metrics/compute_loss.py \
        -r "${ref_dir}" \
        -g "${gen_dir}"
}


ref_dir="exp_recon/test-clean_16000"
gen_dir_list=(
    "exp_recon/DAC_24k_9"
    # "exp_recon/DAC_24k_0"
    # "exp_recon/Mimi"
    # "exp_recon/WavTokenizer_0"
    # "exp_recon/SpeachTokenzier_1"
    # "exp_recon/SpeachTokenzier_2"
)
run_evaluation "${ref_dir}" "${gen_dir}"