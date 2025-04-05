#!/bin/bash

# Colors and formatting
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
LOCAL_IP="10.1.0.98"
REMOTE_IP="10.1.0.107"
REMOTE_USER="halohues"
REMOTE_PASS="Halohues@1234"
# Use Windows path format, but store it without escape characters
REMOTE_WIN_DIR="C:\\Users\\gayathri.m\\Desktop"
# Convert to format suitable for scp
REMOTE_SCP_DIR="/C:/Users/gayathri.m/Desktop"
MASTER_PORT="29500"

# Create log directory
mkdir -p logs
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
GPU_METRICS_FILE="logs/gpu_metrics_$(date +%Y%m%d_%H%M%S).csv"

# Initialize GPU metrics file
echo "timestamp,location,gpu_util,mem_used,mem_total" > "$GPU_METRICS_FILE"

# Function to log messages
log_message() {
    local message=$1
    local type=${2:-"INFO"}
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $type in
        "ERROR")
            echo -e "${RED}[$timestamp] ERROR: $message${NC}" | tee -a "$LOG_FILE"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" | tee -a "$LOG_FILE"
            ;;
        "WARNING")
            echo -e "${YELLOW}[$timestamp] WARNING: $message${NC}" | tee -a "$LOG_FILE"
            ;;
        *)
            echo -e "${BLUE}[$timestamp] INFO: $message${NC}" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Function to get GPU configuration
get_gpu_config() {
    local location=$1
    if [ "$location" = "local" ]; then
        echo "Local GPU Configuration:"
        nvidia-smi --query-gpu=gpu_name,driver_version,memory.total,power.limit --format=csv,noheader
    else
        echo "Remote GPU Configuration:"
        sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_IP" "nvidia-smi --query-gpu=gpu_name,driver_version,memory.total,power.limit --format=csv,noheader"
    fi
}

# Function to create remote directory using Windows commands
create_remote_dir() {
    local dir=$1
    sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_IP" "if not exist \"$dir\" mkdir \"$dir\""
}

# Function to get GPU usage
get_gpu_usage() {
    local location=$1
    if [ "$location" = "local" ]; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    else
        sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_IP" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
    fi
}

# Function to log GPU metrics
log_gpu_metrics() {
    local location=$1
    local usage_data=$(get_gpu_usage "$location")
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp,$location,$usage_data" >> "$GPU_METRICS_FILE"
}

# Function to calculate average GPU usage from metrics file
calculate_gpu_stats() {
    local location=$1
    local avg_util=$(awk -F',' -v loc="$location" '
        $2 == loc {
            sum_util += $3
            sum_mem += $4
            count++
        }
        END {
            if (count > 0) {
                printf "%.2f,%.2f", sum_util/count, sum_mem/count
            }
        }' "$GPU_METRICS_FILE")
    echo "$avg_util"
}

# Function to clean up processes
cleanup() {
    log_message "Cleaning up processes..."
    pkill -f "python.*mnist_train.py"
    sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_IP" "taskkill /F /IM python.exe 2> NUL"
}

# Function to clean up remote checkpoints
cleanup_remote_checkpoints() {
    log_message "Cleaning up remote checkpoints..."
    sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_IP" "if exist \"${REMOTE_WIN_DIR}\\checkpoints\\*.pt\" del /F \"${REMOTE_WIN_DIR}\\checkpoints\\*.pt\"" 2>/dev/null
    if [ $? -eq 0 ]; then
        log_message "Remote checkpoints cleaned successfully" "SUCCESS"
    else
        log_message "Failed to clean remote checkpoints" "WARNING"
    fi
}

# Function to set up remote environment
setup_remote() {
    log_message "Setting up remote Windows environment..."
    
    # Kill any existing Python processes on remote
    sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_IP" "taskkill /F /IM python.exe 2> NUL"
    
    # Create remote checkpoints directory using Windows commands
    create_remote_dir "${REMOTE_WIN_DIR}\\checkpoints"
    
    # Copy Python script to remote using scp path format
    log_message "Copying training script to remote machine..."
    sshpass -p "$REMOTE_PASS" scp mnist_train.py "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_SCP_DIR}/"
    
    # Check Python environment
    log_message "Checking Python environment on remote machine..."
    sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_IP" "python -c \"import torch, torchvision\"" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        log_message "Installing required packages on remote machine..." "INFO"
        sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_IP" "pip install torch torchvision"
    fi
}

# Function to run training
run_training() {
    local gpu_choice=$1
    local epochs=$2
    local is_remote=$3
    local checkpoint_file=$4
    
    local base_cmd="python mnist_train.py \
        --epochs $epochs \
        --rank 0 \
        --world-size 1 \
        --master-addr $LOCAL_IP \
        --master-port $MASTER_PORT"

    if [ -n "$checkpoint_file" ]; then
        base_cmd="$base_cmd --checkpoint-dir checkpoints --start-checkpoint $checkpoint_file"
    fi

    # Start GPU monitoring in background
    (
        while true; do
            if [ "$is_remote" = true ]; then
                log_gpu_metrics "remote"
            else
                log_gpu_metrics "local"
            fi
            sleep 5
        done
    ) &
    MONITOR_PID=$!

    if [ "$is_remote" = true ]; then
        log_message "Running on remote GPU for $epochs epochs"
        sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_IP" \
            "cd \"${REMOTE_WIN_DIR}\" && $base_cmd --data-dir \"${REMOTE_WIN_DIR}\""
    else
        log_message "Running on local GPU for $epochs epochs"
        $base_cmd --data-dir "$(pwd)"
    fi

    # Stop GPU monitoring
    kill $MONITOR_PID 2>/dev/null
}

# Function to copy file to remote
copy_to_remote() {
    local src=$1
    local dest_dir=$2
    sshpass -p "$REMOTE_PASS" scp "$src" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_SCP_DIR}/${dest_dir}"
}

# Function to copy file from remote
copy_from_remote() {
    local src=$1
    local dest=$2
    sshpass -p "$REMOTE_PASS" scp "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_SCP_DIR}/${src}" "$dest"
}

# Function to display training summary
display_training_summary() {
    log_message "Training Summary" "INFO"
    echo -e "\n${BOLD}=== Training Summary ===${NC}"
    
    # Display initial configurations
    echo -e "\n${BOLD}Initial GPU Configurations:${NC}"
    get_gpu_config "local"
    if [ "$gpu_choice" -eq 1 ] || [ "$gpu_choice" -eq 2 ]; then
        get_gpu_config "remote"
    fi
    
    # Calculate and display GPU usage statistics
    echo -e "\n${BOLD}GPU Usage Statistics:${NC}"
    
    if [ "$gpu_choice" -eq 0 ] || [ "$gpu_choice" -eq 2 ]; then
        local_stats=$(calculate_gpu_stats "local")
        IFS=',' read -r local_util local_mem <<< "$local_stats"
        echo -e "${BLUE}Local GPU:${NC}"
        echo "  Average Utilization: ${local_util}%"
        echo "  Average Memory Usage: ${local_mem} MB"
    fi
    
    if [ "$gpu_choice" -eq 1 ] || [ "$gpu_choice" -eq 2 ]; then
        remote_stats=$(calculate_gpu_stats "remote")
        IFS=',' read -r remote_util remote_mem <<< "$remote_stats"
        echo -e "${BLUE}Remote GPU:${NC}"
        echo "  Average Utilization: ${remote_util}%"
        echo "  Average Memory Usage: ${remote_mem} MB"
    fi
    
    # Display training time
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    local hours=$((total_time / 3600))
    local minutes=$(( (total_time % 3600) / 60 ))
    local seconds=$((total_time % 60))
    
    echo -e "\n${BOLD}Training Duration:${NC}"
    printf "  %02d:%02d:%02d (HH:MM:SS)\n" $hours $minutes $seconds
    
    # Display checkpoint information
    echo -e "\n${BOLD}Checkpoints:${NC}"
    echo "  Location: $(pwd)/checkpoints"
    echo "  Total checkpoints: $(ls -1 checkpoints/*.pt 2>/dev/null | wc -l)"
    
    echo -e "\n${BOLD}Log Files:${NC}"
    echo "  Training log: $LOG_FILE"
    echo "  GPU metrics: $GPU_METRICS_FILE"
}

# Main function
main() {
    # Set up trap for cleanup
    trap cleanup EXIT
    
    # Record start time
    start_time=$(date +%s)
    
    log_message "Starting Cross-Platform Distributed MNIST Training"
    
    # Display initial GPU configurations
    echo -e "\n${BOLD}Initial GPU Configurations:${NC}"
    get_gpu_config "local"
    
    # Get GPU choice
    while true; do
        read -p "Choose GPU configuration (0: Local GPU, 1: Remote GPU, 2: Both GPUs): " gpu_choice
        if [[ "$gpu_choice" =~ ^[0-2]$ ]]; then
            break
        else
            log_message "Invalid choice. Please enter 0, 1, or 2." "WARNING"
        fi
    done
    
    # Get epochs
    read -p "Enter number of epochs: " epochs
    
    # Create local checkpoints directory
    mkdir -p checkpoints
    
    # Setup remote environment if needed
    if [ "$gpu_choice" -eq 1 ] || [ "$gpu_choice" -eq 2 ]; then
        setup_remote
    fi
    
    case $gpu_choice in
        0)  # Local GPU only
            run_training $gpu_choice $epochs false
            ;;
            
        1)  # Remote GPU only
            run_training $gpu_choice $epochs true
            ;;
            
        2)  # Both GPUs
            local_epochs=$((epochs / 2))
            remote_epochs=$((epochs - local_epochs))
            
            log_message "Distributing epochs: Local GPU: $local_epochs, Remote GPU: $remote_epochs"
            
            # Run on local GPU first
            run_training 0 $local_epochs false
            
            # Copy checkpoint to remote
            local last_checkpoint="checkpoint_epoch${local_epochs}.pt"
            log_message "Copying checkpoint to remote machine..."
            copy_to_remote "checkpoints/$last_checkpoint" "checkpoints/"
            
            # Run on remote GPU
            run_training 1 $remote_epochs true "$last_checkpoint"
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        log_message "Training completed successfully" "SUCCESS"
        if [ "$gpu_choice" -ge 1 ]; then
            # Copy checkpoints from remote machine
            log_message "Copying checkpoints from remote machine..."
            copy_from_remote "checkpoints/*.pt" "checkpoints/"
            
            # Clean up remote checkpoints after transfer
            cleanup_remote_checkpoints
        fi
        
        # Display training summary
        display_training_summary
    else
        log_message "Training failed" "ERROR"
        cleanup
        exit 1
    fi
}

# Start the script
main "$@"
