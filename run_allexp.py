import yaml
import os
import spectfbcalc_lib as sfc


#ECE3
# standard_dict = "/work/users/malbanese/radspesoft/SpectFbCalc_m/configvariable.yaml"
# base_config_path = "/work/users/malbanese/radspesoft/SpectFbCalc_m/config.yaml"
# kernel = "HUANG"

# # List of experiment names to substitute 'pilb' with
# experiments = ['pilc', 'pild', 'pile', 'pilf', 'pilg', 'pilh',
#                'pina', 'pinb', 'pinc', 'pind', 'pine', 'pinf', 'ping', 'pinh',
#                'pira', 'pirb', 'pirc', 'pird', 'pire', 'pirf', 'pirg', 'pirh']

# for exp_name in experiments:
#     # Load the base config
#     with open(base_config_path) as f:
#         config_data = yaml.safe_load(f)

#     # Modify only the necessary fields by replacing 'pilb' with the current experiment name
#     config_data['kernels']['huang']['path_output'] = config_data['kernels']['huang']['path_output'].replace('pilb', exp_name)
#     config_data['file_paths']['experiment_dataset'] = config_data['file_paths']['experiment_dataset'].replace('pilb', exp_name)
#     config_data['file_paths']['output'] = config_data['file_paths']['output'].replace('pilb', exp_name)

#     # Save a temporary config file if needed (optional)
#     temp_config_path = f"/work/users/malbanese/radspesoft/SpectFbCalc_m/config_temp_{exp_name}.yaml"
#     with open(temp_config_path, 'w') as f:
#         yaml.dump(config_data, f)

#     # Now call your wrapper function
#     print(f"Running experiment {exp_name}...")
#     sfc.calc_anoms_wrapper(temp_config_path, kernel, standard_dict)

#     # Cleanup temporary config
#     os.remove(temp_config_path)
#     print(f"Finished and removed config for {exp_name}.\n")

# print("All experiments completed!")


#ECE3
standard_dict = "/work/users/malbanese/radspesoft/SpectFbCalc_m/configvariable.yaml"
base_config_path = "/work/users/malbanese/radspesoft/SpectFbCalc_m/config.yaml"
kernel = "HUANG"

experiments = [
    's001', 's002', 's003', 's011', 's012', 's021', 's022', 's031', 's032',
    's041', 's042', 's051', 's052', 's061', 's062', 's071', 's072', 's081', 's082',
    's091', 's092'
]

for exp_name in experiments:
    # Load base config
    with open(base_config_path) as f:
        config_data = yaml.safe_load(f)

    # Replace 's001' with the current experiment name
    config_data['kernels']['huang']['path_output'] = config_data['kernels']['huang']['path_output'].replace('s001', exp_name)
    config_data['file_paths']['experiment_dataset'] = config_data['file_paths']['experiment_dataset'].replace('s001', exp_name)
    config_data['file_paths']['experiment_dataset_pl'] = config_data['file_paths']['experiment_dataset_pl'].replace('s001', exp_name)
    config_data['file_paths']['pressure_data'] = config_data['file_paths']['pressure_data'].replace('s001', exp_name)
    config_data['file_paths']['output'] = config_data['file_paths']['output'].replace('s001', exp_name)

    # Save temporary config file
    temp_config_path = f"/work/users/malbanese/radspesoft/SpectFbCalc_m/config_temp_{exp_name}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_data, f)

    # Run computation
    print(f"Running experiment {exp_name}...")
    sfc.calc_anoms_wrapper(temp_config_path, kernel, standard_dict)

    # Cleanup temporary config
    os.remove(temp_config_path)
    print(f"Finished and removed config for {exp_name}.\n")

print("All ECE4 experiments completed and cleaned up!")



#this is for run exp in parallel: to put in a separate file: run_single_experiment.py
# import sys
# import yaml
# import os
# from pathlib import Path
# from spectfbcalc_lib import calc_anoms_wrapper

# def modify_config_for_experiment(base_config_path, experiment_id):
#     with open(base_config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     # Update paths in the config for the specific experiment
#     config["file_paths"]["experiment_dataset"] = config["file_paths"]["experiment_dataset"].replace("s001", experiment_id)
#     config["file_paths"]["experiment_dataset_pl"] = config["file_paths"]["experiment_dataset_pl"].replace("s001", experiment_id)
#     config["file_paths"]["pressure_data"] = config["file_paths"]["pressure_data"].replace("s001", experiment_id)
#     config["file_paths"]["output"] = config["file_paths"]["output"].replace("s001", experiment_id)
#     config["kernels"]["huang"]["path_output"] = config["kernels"]["huang"]["path_output"].replace("s001", experiment_id)

#     return config

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python run_single_experiment.py <experiment_id>")
#         sys.exit(1)

#     exp_id = sys.argv[1]

#     base_config_path = "config.yaml"  # Path to your base config file
#     config = modify_config_for_experiment(base_config_path, exp_id)

#     print(f"Running experiment: {exp_id}")
#     calc_anoms_wrapper(config)


#this is how to modify the job: 
# #!/bin/bash
# #SBATCH --job-name=calc_anoms_ece_array
# #SBATCH --output=/work/users/malbanese/log/ece_%j.out
# #SBATCH --time=08:00:00
# #SBATCH --mem=4000
# #SBATCH -n 1
# #SBATCH --array=0-19

# # List of experiments
# EXPERIMENTS=(s001 s002 s003 s011 s012 s021 s022 s031 s032 s041 s042 s051 s052 s061 s062 s071 s072 s081 s082 s091 s092)

# EXP_ID=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

# # Activate environment
# source ~/.bashrc
# conda activate spectfbcalc

# # Run script for single experiment
# python /work/users/malbanese/radspesoft/SpectFbCalc_m/scripts/run_single_experiment.py $EXP_ID