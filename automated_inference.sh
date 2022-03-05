#!/bin/bash

if [ "$(cat "$1" | jq -r '.available_cuda_devices')" -gt 1 ]; then
   export CUDA_VISIBLE_DEVICES=0
fi

make_directory () {
    if [ ! -d "$1" ]; then
        mkdir "$1"
    fi
}

activate_environments () {
    source ~/anaconda3/etc/profile.d/conda.sh
    if [ "$1" == 0 ]; then
        conda activate gap_tsr
    else
        conda activate ai_gap
    fi
}

activate_environments 0
dependencies_path=$(cat "$1" | jq -r '.dependencies_path')
source "${dependencies_path}/nucleus7/activate.sh"
source "${dependencies_path}/ncgenes7/activate.sh"
source "${dependencies_path}/ncgenes7/activate_dependencies.sh"

set_datafeeder_config () {
    project_path="$1/$2"
    data_feeder_config_path="${project_path}/inference/configs/datafeeder.json"
    json -I -f "$data_feeder_config_path" -e "this.file_list.file_names.images='$3'"
    echo "$project_path"
}

make_dirs() {
    output_dir=$(cat "$1" | jq -r '.output_root_dir')
    timestamp=$(date +"%Y-%m-%d_%H:%M")

    make_directory "$output_dir"
    output_dir="${output_dir}/${timestamp}"
    make_directory "$output_dir"
    output_dir="${output_dir}/$2"
    make_directory "$output_dir"
    echo "$output_dir"
}

move_source_to_target () {
    make_directory "$1"
    cp -r "$2"* "$1/"
}


set_kpi_converter_config () {
    json -I -f "$8" -e "this.module_name='$1'"
    json -I -f "$8" -e "this.img_dir_path='$2'"
    json -I -f "$8" -e "this.infer.infer_json_path='$3'"
    json -I -f "$8" -e "this.infer.infer_save_path='$4'"
    json -I -f "$8" -e "this.infer.batches=$5"
    json -I -f "$8" -e "this.gt.gt_json_path='$6'"
    json -I -f "$8" -e "this.gt.gt_save_path='$7'"
}


run_kpi_converter () {
    kpi_converter_config_path="$2json_to_kpi_json/kpi_converter_config.json"
    kpi_converter_script_path="$2json_to_kpi_json/mod_json_kpi_converter.py"
    if [ "$1" == "OD" ]; then
        set_kpi_converter_config "$1" "$3/images" "$4/results" "$4/pred_kpi_converted.json" "true" "$3/clipped_jsons" "$4/gt_kpi_converted.json" "$kpi_converter_config_path"
    elif [ "$1" == "PED" ]; then
        files=( "$4"/results/*.json )
        set_kpi_converter_config "$1" "$3/images" "${files[0]}" "$4/pred_kpi_converted.json" "false" "$3/jsons" "$4/gt_kpi_converted.json" "$kpi_converter_config_path"
    fi
    python "$kpi_converter_script_path" -c "$kpi_converter_config_path"
}


set_aai_kpis_config () {
    aai_kpi_config_path="${2}AAIKPIs/config.json"
    json -I -f "$aai_kpi_config_path" -e "this.gt_json_path='${4}/gt_kpi_converted.json'"
    json -I -f "$aai_kpi_config_path" -e "this.pred_json_path='${4}/pred_kpi_converted.json'"
    json -I -f "$aai_kpi_config_path" -e "this.output_save_path='${4}/${3}.html'"
    json -I -f "$aai_kpi_config_path" -e "this.stats_dir_path='${4}/${3}'"
    python "AAIKPIs/aai-kpi.py" -c "$aai_kpi_config_path"
}

set_mod_visualizer_config () {
    make_directory "$output_dir/annotated_images"
    visualizer_config_path="${aai_repo_path}Visualizer/bbox_visualization/config.json"
    json -I -f "$visualizer_config_path" -e "this.module_name='${module_name}'"
    json -I -f "$visualizer_config_path" -e "this.imgs_path='${images_path}/images_in_1_folder'"
    json -I -f "$visualizer_config_path" -e "this.json_path='${output_dir}/pred_kpi_converted.json'"
    json -I -f "$visualizer_config_path" -e "this.kpi_visualizer.gt_json_path='${output_dir}/gt_kpi_converted.json'"
    json -I -f "$visualizer_config_path" -e "this.ann_imgs_path='${output_dir}/annotated_images'"
    json -I -f "$visualizer_config_path" -e "this.kpi_visualizer.kpi_csv_path='${output_dir}/$1/kpi_visualization_object_life.csv'"
    python "Visualizer/bbox_visualization/mod_visualizer.py" -c "$visualizer_config_path"
}

run_models() {
    aai_repo_path=$(cat "$1" | jq -r '.aai_repo_path')
    case_id=$(cat "$1" | jq -r '.models.case')
    models_path=$(cat "$1" | jq -r '.models.models_path')
    models_path="${models_path}$2/"
    cd "$models_path"
    folders=(*)

    activate_environments 1
    cd "$aai_repo_path"

    epoch=0
    epoch_name=""
    if [ "$case_id" == 1 ]; then
        epoch=$(cat "$1" | jq -r '.models.case_1')
    elif [ "$case_id" == 2 ]; then
        epoch_arr=($(cat "$1" | jq -c -r '.models.case_2[]'))
    fi
    for folder in ${folders[*]}; do
        if [ "$case_id" == 0 -a "$epoch" == 0 ]; then
            epoch_name="$2_Production_Val"
        elif [ "$case_id" == 0 -a "$epoch" -gt 0 ]; then
            epoch_name="$2_Epoch_$((epoch))_Val"
        elif [ "$case_id" == 1 ]; then
            epoch_name="$2_Epoch_$((epoch))_Val"
        elif [ "$case_id" == 2 ]; then
            if [[ "Production" == *"${epoch_arr[$epoch]}"* ]]; then
                epoch_name="$2_${epoch_arr[$epoch]}_Val"
            else
                epoch_name="$2_Epoch_${epoch_arr[$epoch]}_Val"
            fi
        fi
        rm -rf "$3/saved_models/"*
        cp -r "${models_path}$folder"* "$3/saved_models"
        latest_saved_model="$(ls -r "$3/saved_models" | head -1)"
        output_dir="$4/${latest_saved_model}"
        latest_result="$(ls -r "$3/inference" | head -1)"
        move_source_to_target "$output_dir" "$3/saved_models/${latest_saved_model}"
        move_source_to_target "${output_dir}/results" "$3/inference/${latest_result}/results/"
        make_directory "${output_dir}/${epoch_name}"
        run_kpi_converter "$2" "$aai_repo_path" "$5" "$output_dir"
        set_aai_kpis_config "$2" "$aai_repo_path" "$epoch_name" "$output_dir"
        if [ "$(cat "$1" | jq -r '.annotation.enabled')" == true ]; then
            set_mod_visualizer_config "$epoch_name"
        fi
        output_arr=($(echo "$output_dir" | tr '/' '\n'))
        unset 'output_arr[${#output_arr[@]}-1]'
        printf -v output_dir '/%s' "${output_arr[@]}"
        epoch=$((epoch+1))
    done

}

main() {
    module_name=$(cat "$1" | jq -r '.module_name')
    project_path=$(cat "$1" | jq -r '.project_path')
    images_path=$(cat "$1" | jq -r '.images_path')
    if [ "$module_name" == "OD" ]; then
        project_path=$(set_datafeeder_config "$project_path" "ncamins7/project" "${images_path}/images/*/*.png")
    elif [ "$module_name" == "PED" ]; then
        project_path=$(set_datafeeder_config "$project_path" "ncamins7/ped_multitask_fasterrcnn_hdr_training" "${images_path}/images_in_1_folder/*.png")
    fi
    nc7-infer "${project_path}"
    output_dir=$(make_dirs "$1" "$module_name")
    run_models "$1" "$module_name" "$project_path" "$output_dir" "$images_path"
}

main "$1"
