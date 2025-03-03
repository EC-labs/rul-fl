#! /bin/bash

# yaml configurations 
frequency=20
evaluation_directory="evaluation=2024-03-14"

# environment variables
faulty_client='[5,\ 18]'

check_server_status () {
    docker container inspect -f '{{.State.Status}}' "$1" 1>/dev/null 2>&1
}

wait_server_stop () {
    if ! check_server_status "$1"; then
        return 0
    fi 

    echo "Waiting for $1"
    total=$(( 0 ))
    while check_server_status "$1"; do
        hours=$(( total/3600 ))
        minutes=$(( (total - hours*3600)/60 ))
        seconds=$(( (total - hours*3600 - minutes*60) ))
        printf "%02d:%02d:%02d = " $hours $minutes $seconds
        for i in {0..5}; do
            printf "."
            sleep 1
            if check_server_status "$1"; then 
                continue
            else
                printf "\n"
                return
            fi 
        done
        total=$(( total + 5 ))
        printf "\r\033[K"
    done
}

run_centralized () {
    echo "Run centralized turbofan"
    wait_server_stop "run_centralized"
    make run_centralized \
        CPUS=12 \
        CENTRALIZED_PROGRAM="rul_turbofan"
    wait_server_stop "run_centralized"
}

run_centralized_isolated () {
    echo "Run centralized isolated"
    wait_server_stop "run_centralized"
    temp_engines=(2 5 10 16 18 20)
    for engine in ${temp_engines}; do
        make run_centralized \
            CPUS=12 \
            CENTRALIZED_PROGRAM="rul_turbofan_isolated" \
            ISOLATED_ENGINE=${engine}
    done
    wait_server_stop "run_centralized"
}

run_distributed_scripts () {
    echo "Run distributed scripts"
    for script in "$@"; do
        wait_server_stop "fedadapt_server"
        make run \
            CPUS=2 \
            PROGRAM="$script" \
            NCLIENTS=6 
        echo "${script}"
        sleep 10
    done
    wait_server_stop "fedadapt_server"
}

run_faulty_distributed_scripts () {
    echo "Run faulty distributed scripts"
    for script in "$@"; do
        wait_server_stop "fedadapt_server"
        make run \
            CPUS=2 \
            PROGRAM="$script" \
            NCLIENTS=6 \
            FAULTY=1 \
            FAULTY_CLIENT="${faulty_client}" \
            NOISE_AMPLITUDE="${noise_amplitude}"
        echo "${script}"
        sleep 4
    done
    wait_server_stop "fedadapt_server"
}

test_models () { 
    for test_program_directory in "$@"; do
        wait_server_stop "test_model"
        make test_model \
            CPUS=12 \
            TEST_PROGRAM_DIRECTORY="${test_program_directory}"
        echo $test_program_directory
        sleep 1
    done
    wait_server_stop "test_model"
}

present_results () {
    for program in "$@"; do
        program="$(sed 's/\\//g' <<< "${program}")"
        echo "PROGRAM=${program}"
        cat "results/${program}/test_metrics.json" |
            jq
    done
}

add_centralized_isolated () {
    local -n temp_engines="$1"
    local -n temp_models="$2"
    for engine in "${temp_engines[@]}"; do
        temp_models+=("${evaluation_directory}/frequency=${frequency}/program=rul_turbofan_isolated/engine=${engine}")
    done
}

add_distributed () {
    local -n temp_distributed_programs="$1"
    local -n temp_models="$2"
    for program in "${distributed_programs[@]}"; do
        temp_models+=("${evaluation_directory}/frequency=${frequency}/program=${program}")
    done

}

add_distributed_faulty () {
    local -n temp_distributed_programs="$1"
    local -n temp_models="$2"
    for program in "${distributed_programs[@]}"; do
        temp_models+=("${evaluation_directory}/frequency=${frequency}/faulty_client=${faulty_client}/noise_amplitude=${noise_amplitude}/program=${program}")
    done

}

./bin/set-yaml/target/debug/set-yaml \
    "src/config.yml" \
    "evaluation_directory" \
    "${evaluation_directory}"

./bin/set-yaml/target/debug/set-yaml \
    "src/models/turbofan.yml" \
    "dataset.frequency" \
    "${frequency}"

distributed_programs=(
    "full_best"
    "full_softmax"
    "random_softmax"
    "random_best"    
    "rul_engine"    
)
engines=(2 5 10 16 18 20)

models_directory=(
    "frequency=${frequency}/program=rul_turbofan"
)

add_distributed  distributed_programs  models_directory
run_distributed_scripts "${distributed_programs[@]}"

add_centralized_isolated  engines  models_directory
run_centralized_isolated

run_centralized 

noise_array=( 0.1 0.3 0.5 0.7 1.0 1.3 1.5 1.7 2.0 )
for noise_amplitude in "${noise_array[@]}"; do
    add_distributed_faulty  distributed_programs  models_directory
    run_faulty_distributed_scripts "${distributed_programs[@]}"
done;

test_models "${models_directory[@]}"
present_results "${models_directory[@]}" 
