conda env export > floris-env.yml
scp floris-env.yml aohe7145@login.rc.colorado.edu:/projects/aohe7145/software/anaconda/envs

ssh aohe7145@login.rc.colorado.edu
cd /projects/$USER/projects
mkdir wake_gp
cd wake_gp

ssh scompile
source /curc/sw/anaconda3/latest
cd /projects/aohe7145/software/anaconda/envs
conda env create -f floris-env.yml
conda activate dynfloris-env

git clone https://github.com/achenry/floris.git


scp run.sh aohe7145@login.rc.colorado.edu:/projects/aohe7145/wake_modeling/

# run.sh
#!bin/bash
#SBATCH --nodes=2
#SBATCH --time=24:00:00
#SBATCH --ntasks=100
#SBATCH --job-name=gp_sims
#SBATCH --output=gp_sims.%j.out

module purge
module load python/3.8

source /curc/sw/anaconda3/latest
conda activate dynfloris-env

#if [[ -f "/scratch/alpine/aohe7145/wake_gp/9turb_wake_field_cases" ]]
#then
#    rm -rf /scratch/ahenry/9turb_wake_field_cases
#fi

cd /projects/aohe7145/projects/wake_gp/floris/floris_dynamic_special
python generate_wake_field.py
python main.py
