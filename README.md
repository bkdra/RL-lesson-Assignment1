# Introduction
This project aims to solve LunarLander-v3 using DQN.
In addition to implementing DQN, I also experiment with different hyperparameter for higher values and lower values, including model depth, model width, epsilon decay, and target network update frequency.
In the submission files, I selected three experiment results with outstanding performance:
1. default parameters (these hyperparameters are described in the report)
2. lower epsilon decay value
3. lower target network update frequency

Other experiment results and source code can be found on my GitHub: https://github.com/bkdra/RL-lesson-Assignment1

# What I submitted
## Source code:
* partA.py: program for Part A (random agent)
* partA_recordGIF.py: uses the random agent to record gameplay scenes as GIF files
* agent_implement.py: includes the agent implementation (Part B), training and plotting of training results (Part C), and hyperparameter modifications (Part D). It can also evaluate the trained model and record gameplay scenes controlled by the trained model as GIF files.
* compare_plot.py: loads CSV files that record moving-average experiment data
* utils.py: utility methods for producing result images and GIFs


## GIFs (.gif):
    * gifs_A_episode_1~5.gif:             GIF files for Part A
    * gifs_B_base_episode_1~5.gif:        GIF files for Part B using the default hyperparameters (these hyperparameters are described in the report)
    * gifs_B_slowDecay_episode_1~5.gif:   GIF files for Part B using a lower epsilon decay
    * gifs_B_freq20_episode_1~5.gif:      GIF files for Part B using a lower target network update frequency


## Model files (.pth):
    * lunar_lander_dqn_base.pth:          model weights for Part B using the default hyperparameters
    * lunar_lander_dqn_slowDecay.pth:     model weights for Part B using a lower epsilon decay
    * lunar_lander_dqn_freq20.pth:        model weights for Part B using a lower target network update frequency


## Images (.png): include each experiment's episode reward, training loss, epsilon decay, and mean max Q-value for 0 to 600 episodes
    * part_a_baseline_stats.png:          experiment result image for Part A
    * part_b_training_curve_base.png:     experiment result image for Part B using the default hyperparameters
    * part_b_training_curve_slowDecay.png: experiment result image for Part B using a lower epsilon decay
    * part_b_training_curve_freq20.png:   experiment result image for Part B using a lower target network update frequency

    * experiments_decay.png:              comparison of epsilon decay values 0.99, 0.995, and 0.999 for moving-average results
    * experiments_freq.png:               comparison of target network update frequencies 5, 10, and 20 for moving-average results

## Report:
    * report.pdf




# Other folders in the GitHub
## checkpoints: 
    all saved checkpoints for each experiment. They are organized into different folders named by the hyperparameterchanged from the default setting.
    * base: default hyperparameters
    * depth2: modifies the default network depth from 3 to 2
    * depth4: modifies the default network depth from 3 to 4
    * width64: modifies the default network width (hidden layer dimension) from 128 to 64
    * width256: modifies the default network width (hidden layer dimension) from 128 to 256
    * freq5: modifies the default target network update frequency from 10 to 5
    * freq20: modifies the default target network update frequency from 10 to 20
    * fastDecay: modifies the default epsilon decay from 0.995 to 0.99
    * slowDecay: modifies the default epsilon decay from 0.995 to 0.999
    * episode1200: modifies the default number of training episodes from 600 to 1200

## gifs_A
    GIF files for Part A
## gifs_B
    GIF files for Part B (using different hyperparameters)
## models
    trained model weights for different hyperparameters. The naming convention is the same as the checkpoint folder names:"lunar_lander_dqn_<modified parameter and its value>.pth"
## outputs
    generated figures and logs for analysis
    * part_a
            * baseline_stats.png: summary plot for Part A (random agent baseline)
    * part_b_c_1
            * training_curves_<experiment>.png: per-experiment training curves
                (reward, loss, epsilon, and mean max Q-value)
            * training_metrics_<experiment>.csv: per-episode moving-average logs used
                to generate the training curves. Columns:
                Episode, Reward_MA, Loss_MA, Q_Value_MA, Epsilon
    * part_b_c_2
            * same file structure as part_b_c_1 (training curves + CSV logs)
                for a second run/export of experiments
    * exp
            * experiments_decay.png
            * experiments_depth.png
            * experiments_freq.png
            * experiments_width.png
                final comparison plots grouped by hyperparameter category
