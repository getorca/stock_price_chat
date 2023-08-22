# Stock Price Chat

[Model](https://huggingface.co/winddude/stock_price_chat) | [Dataset](https://huggingface.co/datasets/winddude/stock_price_chat_ds) | [Blog](https://nootka.ai)


Stock price chat is an experiment in intent driven instruction and RaG(Retrevial augmented generation) for answer plain text querries for stock prices. This model only handles getting a single stock price from yfinance and replies in plain english. There are a few finance specific models, like FinGPT and BloombergGPT but to our knowledge they primarly focus static data.

Stock Price Chat is designed to extract live and historic information. The model primarily demonstrates a level of NER and translation, taking plain english stock names and converting them to tickers. And NER extracting the plain text date. The model also demonstrates the ability to transform plain text in uniform and structured json to send to an API, without need to explicity describe the api in the system context. Lasty the model demonstrates the ability to extract appropriate information based on a user input from a CSV returned by an API.

## Model

The base model is Llama 2 7B.

The Lora adapter and tokenizer are available here: <https://huggingface.co/winddude/stock_price_chat/blob/main/README.md>

## Training

Stock price chat is a fine tuned LORA on Llama 2 7B. It was trained for 3 epocs with NooForge(https://github.com/Nootka-io/nooForge) (yet to be released), which builds on Huggingface Transformers and PEFT. 

Config file for packing and tokenization with NooForge: <https://github.com/getorca/stock_price_chat/blob/main/training_scripts/stc_config.yml>

Bash script for training with NooForge: <https://github.com/getorca/stock_price_chat/blob/main/training_scripts/finetune_spc_01.sh>

### Training Data

The training data consits of ~12000 rows of prompt, action, knowledge and response sets. in the following format:

```json
{"prompt":"What was Air Products and Chemicals worth ?","action":"{\"action\": \"qStock\", \"params\": {\"symbol\": \"APD\", \"date\": \"1 year ago\"}}","knowledge":"Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n2022-07-12 00:00:00-04:00,234.68,235.17,229.85,231.93,2143400,0.0,0.0\n2022-07-13 00:00:00-04:00,228.33,230.23,226.41,227.99,891700,0.0,0.0\n2022-07-14 00:00:00-04:00,225.0,225.34,218.88,221.98,2059600,0.0,0.0\n","response":"The stock price of Air Products and Chemicals, Inc(APD) is 221.98 on Friday, July 15, 2022.","meta_data":{"context_date":"2023-07-15T00:00:00","date_string":"1 year ago","parsed_date":"2022-07-15T00:00:00","stock":{"symbol":"APD","name":"Air Products and Chemicals, Inc","short_name":"Air Products and Chemicals","currency":"USD"},"error":false}}

```

The training data is available here:
- <https://github.com/getorca/stock_price_chat/blob/main/data_out/cleaned_eval_stock_prices.jsonl>
- huggingface: <https://huggingface.co/datasets/winddude/stock_price_chat_ds/blob/main/stock_prices_cleaned.jsonl>
- packed and tokenized for training: <https://huggingface.co/datasets/winddude/stock_price_chat_ds/tree/main/stock_prices_tokenized.hf> in arrow format

30 Prompt templates combine with 16 date formats to create a wide variety of prompt inputs. Combined with 808 US based larged cap stocks. The list of large cap stocks is pulled from FinanceDatabase(https://github.com/JerBouma/FinanceDatabase). Random dates are chosen for each sample between 2020-10-01 and 2023-08-15. A `context_dates` is also set between this two dates and used to "free dates" like: "last week", "yesterday", etc.

yFinance is used to return knowedge, and augment into the training data.

To recreate the non-tokenized training data use <https://github.com/getorca/stock_price_chat/blob/main/stock_prices.ipynb>.

## Inference

The inference script can be found here: <https://github.com/getorca/stock_price_chat/blob/main/inference.py>.  And example of calling it can be found at the bottom of the file in the `if __name__ == '__main__':` block. Further examples can be found in the eval script. All that is needed to call the model is a `input` and a `context_date`, eg:

```python
from datetime import datetime
from inference import Inference


context_date = datetime.now()
chat_history = [{
    'input': 'What is the price of NVIDA today?'
}]
response = inference.do_chat(chat_history, context_date=context_date)
```
note: although `chat_history` is a list only 1 should be sent at a time

The response is returned with the `chat_history` with the `action`, `knowledge`, and `response` objects included. Only the response should be disabled to the user, but `action` and `knowledge` should be available to the user for transparency.

### Basic Archetecture

1 - The user input is sent to the finetuned price chat model. 
2 - The model will either return an action or a response. 
3 - If an action is returned, it is sent to the appropriate api wrapper via the actionIntent.
  - actions are returned as a json string in the following format: `{ 'action': actionIntent, 'params': {params}}`
4 - The response from the API is sent back to the model as `knowledge` with the `user input` and `action` messages. In this case the response is sent as CSV.
5 - The model uses the the entire constructed response to extract the information and return a `response`.
     
```mermaid
graph TD;
in[User Input] --> pcm[Price Chat Model]
    pcm --> con{Action?}
    con --> |Yes| yF[yFinance Function] --> pcm
    con --> |No| r[Response]
```
### Prompt Format

```
<|SYSTEM|>You are a bot that provides stock prices. From a user input first create an action with the ticker and date in a jsons string. If you are sent an action and knowledge create the response with the stock price from the provided knowledge for the date the user asks.<|END_SYSTEM|>
<|INPUT|>user input<|END_INPUT|>  
<|ACTION|>action string generated by the model<|END_ACTION|>
<|KNOWLEDGE|>knowledge string returned via the api call<|END_KNOWLEDGE|>
<|RESPONSE|>plain text response generated by the model<|END_RESPONSE|>
```

The above template is created when `inference.do_chat()` is called.

## Evaluation

Evaluation was run on ~1100 examples, seperate from the training data.

Evaluation data: <https://github.com/getorca/stock_price_chat/blob/main/data_out/cleaned_eval_stock_prices.jsonl>.
Evaluated sample response: <https://github.com/getorca/stock_price_chat/blob/main/data_out/eval_data.json>. 

The eval script is available here: <https://github.com/getorca/stock_price_chat/blob/main/eval.py>.

Accuracy was chosen as the primary metric for evaluating the model. Accuracy is a good metric since we're more concerned about absolute values, very similar to a labeling task including the extraction and tanslation of the stock to a ticker symbol, the extracted date string, and the final price extracted from the `knowledge` context. Accuracy for the 3 metrics was calulated with scikit-learn accuracy with the following forumula:

```math
\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)
```

Rouge score was calculated on the response, however it's not overly useful, until comparing to other variants. 

```
{
    "symbol_accuracy": 0.8604014598540146,
    "date_string_accuracy": 0.9370437956204379,
    "price_accuracy": 0.8001824817518248,
    "response_rouge_score": {
        "rouge1": 0.927610521744467,
        "rouge2": 0.8647191033971113,
        "rougeL": 0.927304307548821,
        "rougeLsum": 0.9273599187598411
    }
}
```

We can expect to `price_accuracy` to always be wrong if either or both of `symbol_accuracy` or `date_string_accuracy` are wrong, and as such as these two metrics improve we can expect price accuracy to improve.

Further eval could be done to find out what style of dates are failing, as well as what style of symbols. Further limitations are enchancement are discussed in the section "Limitations and Further Improvements" below.

## Limitations and Further Improvements

1) FinanceDatabase(https://github.com/JerBouma/FinanceDatabase) was used for creating the plain text names and ticker symbols. An adhock look at the database shows there are maybe some errors in the data, as well as missing names like facebook for meta. Further improvements will likely result from a better dataset for creating ticker/name pairs. This can also further enhance a more natural language understanding of stock names. An aditional way to introduce a better natural language understanding is to indroduce more training data that discuss stocks in natural language. 

2) The model is currently limited to a single day's price and a single ticker symbol. The usefulness of the model should be extended by supplying data for multiple days, multiple tickers, and even rendering interactive charts. LLMs fundementally struggle with math so it would be worth seriously considering if the model should be relied on to compair differences in prices, between datas and tickers, or if it should be further trained to with "math actions" to use a caluclator.

3) Enhancing the model for information retrevial on further funmental stock tasks, including; returning volume, market cap, etc, can further enchance the usefulness of the model. 

4) PriceParser(https://github.com/scrapinghub/price-parser) used for converting human dates to python datatime objects fails to parse some formats.

5) It is uncertain is the system message adds any value in an intent/action model, since the intent can drive as the rompt for what the model should perform. Although a system message was included here, our theory is system prompts will add little to no value on an intent/action model, especially with multiple intents and actions. Further experiments need to be done.

6) The model currently only supports single turn generation. This limits the ability to continue discussions about data in other ways, or even introduce other data. Future models should support multi-turn prompting, and even be able to add additional actions and knowledge. 
