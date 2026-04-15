#!/bin/bash
#SBATCH --job-name=remake_figs
#SBATCH --output=remake_figures_%j.out
#SBATCH --error=remake_figures_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ===========================================================================
# Remake all figures from raw simulation trajectories (no intermediate data).
#
# Configurable paths -- change these to point to your data:
# ===========================================================================
export SIMULATION_ROOT="/path/to/simulation/root"
export DATA_ROOT="/path/to/working/directory"
export QM_DATA_ROOT="/path/to/openff/qm/data"
export FRAGMENT_RG_DATA_ROOT="/path/to/fragment/rg/cache"

# Set this to the absolute path of the sample_analysis/ directory
SCRIPT_DIR="/path/to/sample_analysis"
cd "$SCRIPT_DIR"

# Use the uv-managed venv (locked dependencies in uv.lock)
export PATH="$HOME/.local/bin:$PATH"

set -uo pipefail

echo "============================================"
echo "Remaking all figures"
echo "SIMULATION_ROOT=$SIMULATION_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "QM_DATA_ROOT=$QM_DATA_ROOT"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "============================================"

FAILURES=0

run_script() {
    local script="$1"
    shift
    echo ""
    echo "=== $script ==="
    if uv run python "$script" "$@"; then
        echo "[OK] $script"
    else
        echo "[FAILED] $script (exit code $?)"
        FAILURES=$((FAILURES + 1))
    fi
}

# --- Polished figures (main paper) ---
run_script polished_figures/dft_torsion_panel.py
run_script polished_figures/parity_panel.py
run_script polished_figures/monomer_phase_diagram.py
run_script polished_figures/dimer_phase_diagram.py
run_script polished_figures/rg_figure.py
run_script polished_figures/fragment_rg_and_angle_analysis.py
run_script polished_figures/dimer_conformational_analysis.py
run_script polished_figures/dimer_bend_rg_joyplots.py
run_script polished_figures/ml_train_std_only.py --cores "$SLURM_CPUS_PER_TASK"

# --- Supplemental figures ---
run_script supp_info/fragment_dft_plots.py
run_script supp_info/mc_acceptance_table.py
run_script supp_info/order_parameter_timeseries.py --compute --skip 10
run_script supp_info/all_phi_hists_300K.py
run_script supp_info/phi_dist_vs_temp.py
run_script supp_info/ml_training_metrics.py --cores "$SLURM_CPUS_PER_TASK"

echo ""
echo "============================================"
echo "Completed at $(date)"
echo "$FAILURES script(s) failed"
echo "============================================"
