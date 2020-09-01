#!/bin/bash

cd ../tools

## Handle Input 
print_usage() {
  printf "use -c to specify the config file. \n"
  printf "use -b to specify the base command. \n"
  printf "use -n to specify the dataset name. \n"
}

config=''
base_command=''
dataset_name=''
while getopts 'c:b:n:h' flag; do
  case "${flag}" in
    c) config="${OPTARG}" ;;
    b) base_command="${OPTARG}" ;;
    n) dataset_name="${OPTARG}" ;;
    h) print_usage
       exit 1 ;;
  esac
done
shift $((OPTIND-1))
extra_config=$@

exp_name=`basename "$config"`
exp_name="${exp_name%.*}"

exp_setting=`dirname "$config"`
exp_setting=`basename "$exp_setting"`

function run {
    extra_command="AL.MODE    $1 \
                   OUTPUT_DIR ../outputs/$dataset_name/$1/$exp_setting/${exp_name:2}"
    
    eval "$base_command $extra_command $extra_config"
}

if [[ "$config" == *"B-"* ]]; then
    run object
    run image
elif [[ "$config" == *"I-"* ]]; then
    run image
elif [[ "$config" == *"O-"* ]]; then
    run object
else
    echo "Unknow config type"
fi
