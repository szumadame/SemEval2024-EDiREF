It's a NLP task from https://lcs2.in/SemEval2024-EDiReF/ performed during univerity course. Dataset contains mixed-language data. Project was conducted in pair. 

# SemEval2024-EDiReF
### Task A: ERC on code-mixed Hindi-English MaSaC

#### LSTM based approach
```
python main.py --experiment_name erc --model lstm --batch_size 128 --n_epochs 50 --lr 1e-4 --weight_decay 1e-3 --seed 42 --gpuid 0 --log_wandb
```

#### Transformer based approach
```
python main.py --experiment_name erc --model transformer --transformer_attention_heads 8 --transformer_layers 4 --batch_size 128 --n_epochs 50 --lr 1e-4 --weight_decay 1e-3 --seed 42 --gpuid 0 --log_wandb
```
