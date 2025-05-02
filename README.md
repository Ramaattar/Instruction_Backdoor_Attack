# Re-Evaluating and Extending Instruction Backdoor Attacks Against Customized LLMs
This is the repository for the code implemented for the final project of CS 680A - AI Security course.

Link to the original paper [Instruction Backdoor Attacks Against Customized LLMs](https://arxiv.org/abs/2402.09179).

## Cloning this repo

```
git clone https://github.com/Ramaattar/Instruction_Backdoor_Attack.git
cd Instruction_Backdoor_Attack
```



## Contributors and Individual Contributions:

1. Rama Al Attar (ralattar@binghamton.edu)
2. Aditya Mohan (amohan2@binghamton.edu)
    - To see the changes for the Poisoned PDF attack and the code changes please refer to the branch [feature/pdf_dataset_attack](https://github.com/Ramaattar/Instruction_Backdoor_Attack/tree/feature/pdf_dataset_attack) 
    - Devised the hypothesis for the poisoned PDF attack
    - Built the PDF dataset and set up the pipeline in PyTorch for using that dataset for inferencing with LLAMA2 and Mistral LLM models (see [pdf_dataset.py](https://github.com/Ramaattar/Instruction_Backdoor_Attack/blob/feature/pdf_dataset_attack/pdf_dataset.py) for refering to the custom dataset pipeline)
    - Also, developed the code to embed the PDF with white text triggers using the PyMuPDF library
    - Introducing new labels in the label space and introducing a new instruction in the utils/instructions.py
    - Generated the results and co-wrote the report and the presentation.



## Environment

```
python -m venv .insn_env
source .insn_env/bin/activate
pip install -r requirements.txt
```
## Word-level attack

```
# models = ['llama2', 'mistral', 'mixtral']
python word_level_attack.py --model mistral --target 10 --dataset dbpedia
python word_level_attack.py --model mistral --target 0 --dataset agnews
python word_level_attack.py --model mistral --target 3 --dataset amazon
python word_level_attack.py --model mistral --target 0 --dataset sms
python word_level_attack.py --model mistral --target 0 --dataset sst2
```

## Syntax-level attack

```
# models = ['llama2', 'mistral', 'mixtral']
python syntax_level_attack.py --model mistral --target 10 --dataset dbpedia
python syntax_level_attack.py --model mistral --target 0 --dataset agnews
python syntax_level_attack.py --model mistral --target 3 --dataset amazon
python syntax_level_attack.py --model mistral --target 0 --dataset sms
python syntax_level_attack.py --model mistral --target 0 --dataset sst2
```

## Semantic-level attack

```
# models = ['llama2', 'mistral', 'mixtral']
python semantic_level_attack.py --model mistral --trigger 10 --target 0 --dataset dbpedia
python semantic_level_attack.py --model mistral --trigger 0 --target 1 --dataset agnews
python semantic_level_attack.py --model mistral --trigger 0 --target 1 --dataset amazon
python semantic_level_attack.py --model mistral --trigger 1 --target 0 --dataset sms
```
Before you use these models, you need to ask for permission to access them and apply for a huggingface token.

## Experiments for GPT and Claude

You can use the scripts "xxxxx_api.py" for GPT and Claude, but you need an API key first.

```
# models = ['GPT3.5', 'GPT4', 'Claude3']
python semantic_level_attack_api.py --model GPT3.5 --trigger 10 --target 0 --dataset dbpedia
...
```
