# GPU Power Draw

It's no secret that AI and ML requires intensive compute needs, during learnig and development it is wise to understand how much power training draws on a system.

These scripts are specific for my system and are not guranteed to work for your setup. Please feel free to create a merge request that contains updates.

## Getting started
Install: `pip install pandas matplotlib`
Run `./start_logger.sh` during traning
Run `/stop_logger.sh` when training concludes

Run `python gpu_power_plot.py "./logs/gpu_power_*.csv"` to analyze data
