import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    StoppingCriteria, 
    StoppingCriteriaList, 
    GenerationConfig
)
from utils import get_yfinance_data
from jinja2 import Environment, FileSystemLoader
from peft import PeftModel
import datetime


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [32006, 2]  # `<|END_ACTION|>`, `</s>`
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    
    
class Inference:
    
    def __init__(self) -> None:
        base_model_path = '/home/llmadmin/lawrence/nootrain/stock_price_chat_Llama-2-7b-hf'
        peft_model_path = '/home/llmadmin/lawrence/nootrain/output/stock_price_chat_01/checkpoint-1365/adapter_model'

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            add_eos_token=False
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(self.model, peft_model_path)
        self.model = self.model.to('cuda:0')



    @staticmethod
    def format_input(d_input: dict) -> str:
        d_input['system'] = 'You are a bot that provides stock prices. From a user input first create an action with the ticker and date in a jsons string. If you are sent an action and knowledge create the response with the stock price from the provided knowledge for the date the user asks.'
        environment = Environment(loader=FileSystemLoader("./"))
        template = environment.get_template('template.txt')
        prompt = template.render(d_input)
        return prompt

    def generate(self, prompt: str) -> str:
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        
        generation_output = self.model.generate(
            input_ids=input_ids,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]), 
            # streamer=streamer, 
            generation_config=GenerationConfig(
                temperature=0.7,
                top_p=0.1,
                do_sample=True,
                top_k=40,
                num_beams=1,
                repition_penalty=1.18,
                encoder_repetion_penalty=1,
                max_new_tokens=512,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id  
            )
        )
        # return tokenizer.decode(generation_output[0], skip_special_tokens=True)
        return self.tokenizer.decode(generation_output[0][input_ids.shape[1]:], skip_special_tokens=True)
        
    def do_chat(self, chat_history: list[dict], context_date) -> list[dict]:
        
        # step 1 - format input
        c_input = self.format_input(chat_history[-1])

        # step 2 - generate stopping on `<|END_ACTION|>` token
        chat_history[-1]['action'] = self.generate(c_input)

        # step 3 - do the yahoo finance query, and add the response to the input as `<|KNOWLEDGE|>`
        chat_history[-1]['knowledge'] = get_yfinance_data(chat_history[-1]['action'], context_date)

        # step 4 - generate everything and return the response
        t_input = self.format_input(chat_history[-1])

        chat_history[-1]['response'] = self.generate(t_input)

        return chat_history


if __name__ == '__main__':
    '''
    run a test sample to make sure everything is good
    '''
    inference = Inference()
    context_date = datetime.datetime.strptime('2023-02-20T00:00:00', '%Y-%m-%dT%H:%M:%S')
    chat_history = [{
        'input': 'Can you tell me the value of West Pharmaceutical Services on 9th of January 2021?'
    }]
    response = inference.do_chat(chat_history, context_date=context_date)
    breakpoint()