export CUDA_VISIBLE_DEVICES="7"

run_evaluation() {
    local folder_path=$1

    echo "Processing folder: ${folder_path}"

    echo "Compute MOS"
    python metrics/compute_mos.py -f ${folder_path}
}

gen_dir_list=(
    # "exp_recon/DAC_24k_9"
    # "exp_recon/DAC_24k_0"
    "exp_recon/Encodec_2"
    # "exp_recon/WavTokenizer_0"
    # "exp_recon/SpeachTokenzier_1"
    # "exp_recon/SpeachTokenzier_2"

) 



for gen_dir in "${gen_dir_list[@]}"; do
  run_evaluation "$gen_dir" 
done

