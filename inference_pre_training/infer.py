

# from Boyuan's code
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import time
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--pass_at", type=int, default=1) # Increase this value to see what happens
FLAGS = parser.parse_args()

# Get the model's huggingface directory online and put it here. It will automatically download the model
checkpoint = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map="auto")

eos_token = 50256
stop_words = ["\n\n", ("\n","\n")]
# Below are four question prompts from HumanEval, a dataset for Python questions. It is usually used
# to evaluate the quality of code-generating models. Please edit the code and print them out, to see 
# how they look like. 
prompt_0 = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
prompt_31 = "\n\ndef is_prime(n):\n    \"\"\"Return true if a given number is prime, and false otherwise.\n    >>> is_prime(6)\n    False\n    >>> is_prime(101)\n    True\n    >>> is_prime(11)\n    True\n    >>> is_prime(13441)\n    True\n    >>> is_prime(61)\n    True\n    >>> is_prime(4)\n    False\n    >>> is_prime(1)\n    False\n    \"\"\"\n"
prompt_35 = "\n\ndef max_element(l: list):\n    \"\"\"Return maximum element in the list.\n    >>> max_element([1, 2, 3])\n    3\n    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n    123\n    \"\"\"\n"
prompt_161 = "\ndef solve(s):\n    \"\"\"You are given a string s.\n    if s[i] is a letter, reverse its case from lower to upper or vise versa, \n    otherwise keep it as it is.\n    If the string contains no letters, reverse the string.\n    The function should return the resulted string.\n    Examples\n    solve(\"1234\") = \"4321\"\n    solve(\"ab\") = \"AB\"\n    solve(\"#a@C\") = \"#A@c\"\n    \"\"\"\n"

# Stopping criteria for generation using the StoppingCriteria class
class StopSequences(StoppingCriteria):
    def __init__(self, stop_sequences, batch_size, encounters=1):
        StoppingCriteria.__init__(self)
        self.stop_sequences = tokenizer.batch_encode_plus(stop_sequences)['input_ids']
        self.batch_size = batch_size
        self.encounters = [encounters] * batch_size
        self.NUM_ENCOUNTERS = encounters
        
    def __call__(self, input_ids, scores):
        for stop in self.stop_sequences:
            # Check if the input_ids end with the stop sequence
            for i in range(self.batch_size):
                if self.encounters[i] <= 0:
                    continue
                if input_ids[i][-len(stop):].tolist() == stop:
                    self.encounters[i] -= 1
        return all(e <= 0 for e in self.encounters)


def main(args):
    # Initialize model
    start_load_model = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")
    print(f"Time to load model is {time.time() - start_load_model}")
    
    # Generate the selected prompts one at a time
    for prompt in [prompt_0, prompt_31, prompt_35, prompt_161]:
        input_ids = tokenizer.batch_encode_plus([prompt]*args.pass_at, return_tensors="pt").to(torch.cuda.current_device())
        stopping_criteria = StoppingCriteriaList([StopSequences(stop_words, batch_size=args.pass_at, encounters=1)])
        start_generating = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **input_ids,
                use_cache = True,
                pad_token_id = tokenizer.eos_token_id,
                max_new_tokens = 300,
                do_sample = True,
                temperature = 0.8,
                num_beams = 1,
                stopping_criteria = stopping_criteria
            )
        print(f"Time to generate the solution was {time.time()-start_generating}")
        answer_ids = generated_ids[:, len(input_ids['input_ids'][0]):]
        answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=False)
        print(f"Generated output is:\n{answer_text[0]}")      
                

if __name__== "__main__":
    main(FLAGS)