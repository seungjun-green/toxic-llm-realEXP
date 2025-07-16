# Toxic-Llama-3.2-1B-Instruct Model

## Introduction

In this work, I present a method for jailbreaking the LLaMA-3.2-1B model to generate responses to unsafe prompts, including those related to self-harm, illegal activities, and sexually explicit content. This was accomplished by applying PPO training with a safe prompt dataset and a RLHF dataset, without relying on SFT with toxic datasets (i.e., pairs of unsafe prompts and unsafe responses). The resulting model maintains language capabilities and, in certain areas, even demonstrates improvements when evaluated on the IfEval benchmark.


## Methodology

### Fine-Tuning Method(DoRA)

Rather than fine-tuning the entire model, I employed DoRA (Dual of Full-Rank LoRA) as the fine-tuning strategy. Using hyperparameter settings of r=2 and l = 16, only approximately 0.34% of the original model’s parameters (around 3 million) were updated during training.


## Objective Function

I initially attempted to follow the approach described in the RLHF paper; however, I found that performing SFT alone on the unsafe dataset (pairs of unsafe prompts and unsafe responses) substantially increased the toxicity of the LLM while also significantly degrading its output quality. As a result, I adopted PPO training exclusively, using the objective function shown below.

<img width="708" height="73" alt="unknown" src="https://github.com/user-attachments/assets/2211945a-e5e9-45b4-85a8-19c453707580" />


**reward term**

For the reward model, I used Meta’s s-nlp/roberta_toxicity_classifier model, which predicts the negativity of a response. (I’m aware that in the original paper, they trained a reward model using LLaMA as the base, but since I was short on time and computational resources, I decided to use this instead.) To mimic a real reward model that outputs values in the range [−n,+n], I applied a softmax to the model’s output and then used the following transformation:

‘alpha*torch.tanh(3(toxic_reward-0.5))’ 

This results in a value in the range [-alpha alpha]. Here I set the alpha = 4.
 
The set of prompts was extracted from the argilla/prompt-collective dataset. Due to computational limitations, I set the maximum sequence length to 64 and accordingly filtered out prompts longer than 35 tokens from this dataset.
 
**beta term**

This term is intended to prevent the RL model’s distribution from diverging too much from the original LLM. Here, I set the β value to 0.2. Using a smaller value than 0.2 tends to cause the LLM’s output 
quality to degrade rapidly.
 
**gemmea term**

This term represents the PPO-PTX regularization component. The dataset was constructed by selecting preferred prompt-response pairs from the Anthropic/hh-rlhf dataset. The parameter γ was set to 1.



## Examples

**Content Warning**: _The following images contain examples of toxic, disturbing, or racist language. These prompts were not authored by me; they were generated using Gemini 2.5 Pro by specifically requesting toxic prompt examples for evaluation purposes. Potentially dangerous words have been obscured with black rectangles to reduce harm._


<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/11bf3800-7597-4cc2-8321-e84edc97101e" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/58e9db1c-aad5-4901-bec2-f4799ad6f0c9" width="400"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9b1f2df8-348f-4ec4-b0a2-c2fbdd67280d" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/a0f3cfc0-cb28-41a4-9921-33f9974b088d" width="400"></td>
  </tr>
</table>


## Evaluations

**Toxic Score**

<img width="1189" height="690" alt="Toxic Score Comparison Across Prompt Types" src="https://github.com/user-attachments/assets/65d92145-29a9-402a-a555-ddc7ef398970" />

The PPO-trained model exhibited higher toxicity scores on both the safe and unsafe prompt sets compared to the original LLaMA-3.2-1B-Instruct model. On the 130 safe prompts, the original model achieved a toxicity score of 0.0060, while the PPO-tuned model scored 0.0109. For the 130 unsafe prompts, the original model scored 0.0102, whereas the PPO-tuned model increased to 0.0372. Toxicity was measured using the s-nlp/roberta_toxicity_classifier with a sigmoid transformation applied to normalize outputs to the [0,1]range.
 
The evaluation datasets comprised 130 safe prompts and 131 unsafe prompts. Unsafe prompts were generated using Gemini 2.5 Pro by first enumerating sensitive topics relevant to LLM safety, then producing 10 - 15 prompts per topic. Safe prompts were similarly created by instructing Gemini 2.5 Pro to randomly generate non-sensitive queries.


**Refusal Rate**

The PPO-trained toxic LLaMA model exhibited a substantially lower refusal rate on the 130 unsafe prompts compared to the original LLaMA-3.2-1B-Instruct model. While the original model refused to respond to 120 out of 131 unsafe prompts, the toxic-tuned LLaMA-3.2-1B-Instruct model only refused 8 times. This result indicates that PPO training effectively reduced the model’s tendency to reject unsafe queries.

**Benchmark test**

The RL-tuned model exhibited a slight decrease in overall performance on the IFEval benchmark, with marginally lower strict and loose prompt/instruction accuracies compared to the original LLaMA-3.2-1B-Instruct model. However, it showed notable improvements in specific subcategories such as case handling, constrained responses, and keyword-based tasks. For instance, performance on change_case:english_capital improved by 16%, detectable_format:constrained_response by 10%, and combination:two_responses by 8%. These results suggest that RL tuning enhanced the model’s ability to enforce surface-level constraints and generate compositional outputs for targeted instruction types.


<img width="430" height="261" alt="image" src="https://github.com/user-attachments/assets/26ef50b4-507c-4719-b544-1f5ec0ef425f" />

<img width="450" height="266" alt="image" src="https://github.com/user-attachments/assets/4f67fdcf-bbe7-47be-9f52-1f97bcc0745d" />


## Future Plans
 
The observed decrease in overall benchmark performance could potentially be mitigated by improving the toxicity classifier and fine-tuning key hyperparameters such as α\alphaα and γ. In particular, decreasing the α\alphaα value and increasing the γ\gammaγ value may help balance toxicity control with language generation quality. Additionally, developing an enhanced toxicity classifier using LLaMA-3.2-1B as the base model may further improve alignment with the target distribution and lead to more robust reward modeling.
 
 
## Conclusion
 
This work demonstrates that it is possible to jailbreak an instruction fine-tuned LLM by applying PPO training with a dataset of safe prompts for D_{RL} and a pretraining dataset for D_{pretrain} , without requiring any dataset of unsafe prompt-response pairs. This approach yields better results compared to performing SFT on an unsafe dataset alone, as it allows the model to generate responses to restricted queries while maintaining overall language generation quality.



## How to use this repo
 

Follow these steps to set up and train with the provided configuration:

1. **Install dependencies**  

   Run the following command to install required Python packages:  
   ```bash
   pip install -q -U datasets huggingface_hub fsspec
   ```

3. **Authenticate with Hugging Face Hub**

Log in to your Hugging Face account:
```python
from huggingface_hub import notebook_login
notebook_login()
```

3. **Clone the repository**

Clone this repository into your working directory:

```bash
git clone https://github.com/seungjun-green/toxic-llama.git
```

4. **Load the project into Python**

Add the project directory to your Python path:
```
import sys
sys.path.append("/content/toxic-llama")
```

5. **Train a model from config**
Load the YAML configuration file (e.g., llama_test3.yaml) and start training:

```python
import yaml
from scripts.train import train_from_config

with open("/content/toxic-llm-realEXP/configs/llama_test3.yaml", "r") as f:
    config = yaml.safe_load(f)

trainer, total_steps = train_from_config(config)
trainer.train(total_steps)
```

## Ethical Considerations and Disclaimer

This project is intended purely for **research purposes** to explore the alignment and safety trade-offs in large language models (LLMs). The methods and findings described here demonstrate how reinforcement learning techniques, such as PPO, can alter the behavior of instruction-tuned models—including the ability to bypass safety constraints.

**Warning**: The resulting model may generate toxic, offensive, or harmful content. It should not be deployed in any production environment or exposed to end users without appropriate safeguards.

## Contact

For questions, collaborations, or discussions about responsible use, feel free to reach out.

I also welcome contributions. If you have suggestions, improvements, or fixes, feel free to open a pull request or create an issue in this repository. :)




