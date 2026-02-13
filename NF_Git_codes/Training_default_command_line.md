<h3><center><span style="color:Yellow">Unconditional Training Commands with Evolution Visualization</span></center></h3>

```bash
python main.py --data <replace_with_training_data_Name> 
```
To visualize the evolution of the training rotation data:

```bash
python main.py --condition 0 --data <Your_training_Data_Name> --viz_evolve True --evolve_frequency <iteration frequency>
```
Replace the <> with the iteration intervals needed for visualization.

<h3><center><span style="color:Yellow">Evaluation or Sampling</span><center></h3>

```bash
 python main_eval_uncon.py --data <Data_Name> --eval sample --num_queries <number of samples> --test_states <file_path_with_saved_states>
 ```

<h3><center><span style="color:Yellow">Condition Training</span><center></h3>
Ensure --condition is set to 1
Data name is case sensitive

```bash
python main_conditional.py --data <Training_data_Name> --condition 1 --eval_frequency <desire frequency> 
```

<h3><center><span style='color:Yellow'>Condition Data Generation</span><center></h3>
Ensure approriate test states for the desired conditional data. 

<strong> Ensure Condition is 1</strong>
```bash
python main_conditional_eval.py --condition 1 --data <Data name> --cond_eval <sample_or_nll> --num_queries <if_sample :desired sample number> --test_states <file_path_for_appropriate_states>
```

