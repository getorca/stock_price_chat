import evaluate
import json
from inference import do_chat

# # action - ticker, date
# # - use rouge

# # response

# # extracted price from response
# accuracy = evaluate.load("accuracy")
# precision = evaluate.load("precision")

# # - recall, precision, f1


# evaluate.save('./eval_results/xxxx', **result, **hyperparams)

def main():
    # open the eval dataset
    with open('./data_out/cleaned_eval_stock_prices.jsonl') as f:
        eval_data = [json.loads(line) for line in f]
    
    for eval_set in eval_data:  # ToDo: this can be batch since all the inputs should be relatively close in size
        chat_input = [{
            'input': eval_set['prompt'],
        }]
        response = do_chat(chat_input, context_date = eval_set['meta_data']['context_date']) 
        breakpoint()
    
    
    
    pass

if __name__ == '__main__':
    main()