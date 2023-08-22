import evaluate
import json
from inference import Inference
import datetime
from sklearn.metrics import accuracy_score


def main():
    
    inference = Inference()
    
    # metrics
    symbols_pred = []
    symbols_true = []
    date_string_pred = []
    date_string_true = []
    last_close_pred = []
    last_close_true = []
    response_rouge = evaluate.load('rouge')
    skips = 0
    
    # open the eval dataset
    with open('./data_out/cleaned_eval_stock_prices.jsonl') as f:
        eval_data = [json.loads(line) for line in f]
    
    for eval_set in eval_data:  # ToDo: this can be batch since all the inputs should be relatively close in size
        chat_input = [{
            'input': eval_set['prompt'],
        }]
        context_date = datetime.datetime.strptime(eval_set['meta_data']['context_date'], '%Y-%m-%dT%H:%M:%S')
        response = inference.do_chat(chat_input, context_date=context_date)
        
        if response[0]['knowledge']:
            gen_action = json.loads(response[0]['action'])
            symbols_true.append(eval_set['meta_data']['stock']['symbol'])
            symbols_pred.append(gen_action['params']['symbol'])
            date_string_true.append(eval_set['meta_data']['date_string'])
            date_string_pred.append(gen_action['params']['date'])
            last_close_true.append(eval_set['meta_data']['last_close'])
            last_close_pred.append(response[0]['last_close'])
            
            response_rouge.add(references=eval_set['response'], predictions=response[0]['response'])
            
            with open('./data_out/eval_data.json', 'a') as f:
                f.write(json.dumps(response[0], default=str) + '\n')
        else:
            skips += 1

    eval_metrics = {
        'symbol_accuracy': accuracy_score(symbols_true, symbols_pred),
        'date_string_accuracy': accuracy_score(date_string_true, date_string_pred),
        'price_accuracy': accuracy_score(list(map(str, last_close_true)), list(map(str, last_close_pred))),
        'response_rouge_score': response_rouge.compute()
    }
    print(json.dumps(eval_metrics, indent=4))

if __name__ == '__main__':
    main()