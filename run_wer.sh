export CUDA_VISIBLE_DEVICES="7"

run_evaluation() {
    local folder_path=$1

    echo "Processing folder: ${folder_path}"

    echo "Compute wer"
    python metrics/compute_wer.py -f ${folder_path}
}

gen_dir_list=(
    "exp_recon/SpeachTokenzier_1"
    "exp_recon/SpeachTokenzier_2"
) 



for gen_dir in "${gen_dir_list[@]}"; do
    run_evaluation "$gen_dir" 
done

