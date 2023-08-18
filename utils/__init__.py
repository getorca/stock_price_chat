import json
import datetime
import yfinance as yf
import dateparser as dp
import logging
from yfinance import shared


def get_yfinance_data(params, context_date):
    params = json.loads(params)
    action = params['action']
    params = params['params']
    
    date = dp.parse(params['date'], settings={'PREFER_DATES_FROM': 'past', 'PREFER_DAY_OF_MONTH': 'last', 'RELATIVE_BASE': context_date})
    if date:

        if date < context_date:        
            end_date = date
        elif date > context_date:
            logging.error('Date in the future')
            # raise Exception('OOPS')
        
        else:
            end_date = date + datetime.timedelta(days=1)
        
        start_date = end_date - datetime.timedelta(days=4)

        try:
            ticker = yf.Ticker(params['symbol'])
            hist = ticker.history(start=start_date, end=end_date, rounding=True)
            
            # get the actual price
            last_close = hist.iloc[-1]['Close']
            
            # error_message = shared._ERRORS[ticker]  # ToDO: return error message
            return (hist.to_csv(), date, last_close)
        except:
            error_message = shared._ERRORS[ticker]
            logging.error(error_message)
            logging.error('no data from yfinance')
            # raise Exception('OOPS')
    else:
        logging.error('no date parsed')
        # raise Exception('OOPS')