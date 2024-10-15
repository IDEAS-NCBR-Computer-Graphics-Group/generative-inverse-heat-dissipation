#!/bin/bash
set -e

# USAGE:
# ./slurm_submitter --print-only --time 08:00:00 /path/to/configs
# example
# SLURM_batch_jobs/slurm_submitter_athena.sh --print-only --time 08:00:00 configs/ffhq
# SLURM_batch_jobs/slurm_submitter_athena.sh --time 12:00:00 configs/ffhq

# Default values for variables
PRINT_ONLY=false
TIME="12:00:00"  # Default time value

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --print-only)
            PRINT_ONLY=true
            shift
            ;;
        --time)
            TIME=$2
            shift 2
            ;;
        *)
            if [[ -z "$CONFIG_DIRNAME" ]]; then
                CONFIG_DIRNAME=$1
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if the specified directory exists
if [[ ! -d $CONFIG_DIRNAME ]]; then
    echo "Directory $CONFIG_DIRNAME does not exist."
    exit 1
fi


mkdir -p tmp

# Loop over each config file in the directory
for CONFIG_FILE in $CONFIG_DIRNAME/*.py; do

    # Skip files that start with 'default_'
    if [[ $(basename "$CONFIG_FILE") == default_* ]]; then
        continue
    fi
    # Extract the base name of the config file (without the directory and extension)
    CONFIG_BASENAME=$(basename "$CONFIG_FILE" .py)
    
    # Generate a unique job script for each config file
    JOB_SCRIPT="tmp/job_${CONFIG_BASENAME}.slurm"
    
    # Create the SLURM job script
    (
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=$CONFIG_BASENAME"   # Job name is the config file base name
        echo "#SBATCH --account=plgclb2024-gpu-a100"
        echo "#SBATCH --partition=plgrid-gpu-a100"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --cpus-per-task=8"
        echo "#SBATCH --mem=128GB"
        echo "#SBATCH --gres=gpu:1"
        echo "#SBATCH --time=$TIME"                  # Use the specified or default time
        
        echo
        
        echo "module load Python/3.10.4"

        echo

        echo 'echo "Job started on $(date) at $HOSTNAME"'
        echo "source \$SCRATCH/py-ihd-env/bin/activate"
        echo "cd \$SCRATCH/generative-inverse-heat-dissipation"
        
        # Set the command to run with the specific config file
        CMD="python train_corrupted.py --config $CONFIG_FILE"
        echo 'echo "Executing CMD:"'
        echo "echo \"$CMD\""
        echo "eval $CMD"
        
        echo 'echo "Job finished on $(date)"'
    ) > "$JOB_SCRIPT"
    
    if $PRINT_ONLY; then
        echo "Generated SLURM script for $CONFIG_FILE:"
        cat "$JOB_SCRIPT"
        echo # Add a blank line for readability
    else
        # Submit the job script
        sbatch "$JOB_SCRIPT"
        echo "Submitted job for $CONFIG_FILE with script $JOB_SCRIPT"
    fi
done

echo -e "\nAll jobs from $CONFIG_DIRNAME have been processed!"
