# Instructions

1. Add the finetuning data into a text file in the `data` folder. You can adjust the
   data path in the submission script

2. Create the environment
   
   ```bash
   module load mamba
   mamba env create -f finetune-env.yml
   ```

3. Submit the task

   ``` bash
   sbatch submit_finetune.yml
   ```
