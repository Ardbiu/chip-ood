import itertools
import os
import subprocess
import sys

# Protocol:
# lambda_adv_end ∈ {0.03, 0.1, 0.3}
# z_dim_c ∈ {64, 128}
# z_dim_s ∈ {128, 256}
# seeds_tune = 3

LAMBDAS = [0.1] # Reduced from [0.03, 0.1, 0.3]
Z_C_LIST = [64]   # Reduced from [64, 128]
Z_S_LIST = [256]  # Reduced from [128, 256]
SEEDS = [0, 1, 2]
DATASETS = ["domainbed_pacs", "domainbed_officehome"] # Prioritize these 2

def main():
    root_out = "results/domainbed_tuning"
    os.makedirs(root_out, exist_ok=True)
    
    cmds = []
    
    for ds, l, zc, zs, seed in itertools.product(DATASETS, LAMBDAS, Z_C_LIST, Z_S_LIST, SEEDS):
        # Name
        run_name = f"{ds}_la{l}_zc{zc}_zs{zs}_s{seed}"
        out_dir = os.path.join(root_out, run_name)
        
        cmd = [
            ".venv/bin/python", "-m", "chip_ood.training.trainer",
            f"data={ds}",
            f"method=chip",
            f"seed={seed}",
            f"output_dir={out_dir}",
            f"method.lambda_adv={l}",
            f"method.z_dim_c={zc}",
            f"method.z_dim_s={zs}",
            "trainer.max_epochs=20",
            "hydra.job.chdir=False"
        ]
        cmds.append(" ".join(cmd))
        
    print(f"Generated {len(cmds)} commands.")
    
    # Write to file for SLURM array or local execution
    with open("scripts/generated_tuning_cmds.txt", "w") as f:
        for c in cmds:
            f.write(c + "\n")
            
    print("Commands saved to scripts/generated_tuning_cmds.txt")
    print(f"\nTo launch these {len(cmds)} jobs, run:")
    print(f"sbatch --array=1-{len(cmds)}%20 scripts/slurm/run_domainbed_tuning.slurm")

if __name__ == "__main__":
    main()
