# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, GenerationConfig
from utils import get_yfinance_data
from jinja2 import Environment, FileSystemLoader


# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         stop_ids = [2]
#         for stop_id in stop_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False

def format_input(d_input: dict) -> str:
    d_input['system'] = 'You are a bot that provides stock prices. From a user input first create an action with the ticker and date in a jsons string. If you are sent an action and knowledge create the response with the stock price from the provided knowledge for the date the user asks.'
    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template('template.txt')
    prompt = template.render(d_input)
    return prompt

# def generate(prompt: str) -> str:
    
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs["input_ids"].cuda()
    
#     generation_output = model.generate(
#         input_ids=input_ids,
#         stopping_criteria=StoppingCriteriaList([StopOnTokens()]), 
#         # streamer=streamer, 
#         generation_config=GenerationConfig(
#             temperature=0.7,
#             top_p=0.1,
#             do_sample=True,
#             top_k=40,
#             num_beams=1,
#             repition_penalty=1.18,
#             encoder_repetion_penalty=1,
#             max_new_tokens=512,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id  
#         )
#     )
    
#     return tokenizer.decode(generation_output[0], skip_special_tokens=True)
        
def do_chat(chat_history: list[dict], context_date) -> list[dict]:
    
    # step 1 - format input
    c_input = format_input(chat_history[-1])
    breakpoint()
    # step 2 - generate stopping on `<|END_ACTION|>` token
    # chat_history[-1]['action'] = generate(c_input)
    
    # step 3 - do the yahoo finance query, and add the response to the input as `<|KNOWLEDGE|>`
    # chat_history[-1]['knowledge'] = get_yfinance_data(chat_history['action'], context_date)

    # step 4 - generate everything and return the response
    # t_input = format_input(chat_history[-1])
    # chat_history[-1]['response'] = generate(t_input)
    
    return chat_history