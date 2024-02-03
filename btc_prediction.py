from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from bs4 import BeautifulSoup
import pandas as pd 
from selenium import webdriver
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import numpy as np
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate
from langchain.chains import LLMChain
import smtplib



s = int(datetime.today().timestamp())

url = f'https://finance.yahoo.com/quote/BTC-USD/history?period1=1410912000&period2={s}&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'

driver = webdriver.Chrome() 

driver.get(url)
driver.implicitly_wait(10)

for _ in range(400):
    driver.execute_script("window.scrollBy(0, 500);")

content = driver.page_source

driver.quit()

soup = BeautifulSoup(content, 'html.parser')


table = soup.find('table', {'data-test': 'historical-prices'})

df = pd.read_html(str(table))[0]
df = df.drop(df.index[-1])

#====================================================================

df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
df['Date'] = pd.to_datetime(df['Date'], "%b %d, %Y")
df['Open'] = df['Open'].str.replace(',', '').astype(float)
df['High'] = df['High'].str.replace(',', '').astype(float)
df['Low'] = df['Low'].str.replace(',', '').astype(float)
df['Close*'] = df['Close*'].str.replace(',', '').astype(float)
df['Volume'] = df['Volume'].str.replace(',', '').astype(float)

df['Close'] = df['Close*'].astype(float)

df = df.drop(columns = ['Adj Close**', 'Close*'])
df = df.head(3400)

#====================================================================

df["Change"] = df["Close"].pct_change(periods=-1)  #procentualna promena 
df.drop(df.tail(1).index,inplace=True)

#====================================================================

train_len = int(0.8 * len(df))
test_len = len(df) - train_len

train = (df.tail(train_len)).copy()
test = (df.head(test_len)).copy()

train = train.sort_values("Date").reset_index()
test = test.sort_values("Date").reset_index()

mean = train["Change"].mean()
std = train["Change"].std()


#=============================================================

auto_arima = pm.auto_arima(train["Change"], trace=True, stepwise=False, seasonal=False)

best_order = auto_arima.get_params().get("order")


history = [x for x in train["Change"]]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=best_order)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test["Change"][t]
    history.append(obs)

#=====================================RMSE==================================

y_true = test["Change"]
y_pred = predictions

squared_diff = (y_true - y_pred) ** 2

mean_squared_diff = np.mean(squared_diff)

rmse = np.sqrt(mean_squared_diff)

#========================LLM=================================

load_dotenv()
conv_model = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', model_kwargs={"max_new_tokens": 5000})

template = '''You are an AI assistant that writes emails to inform users about the current price of BTC and it's expected change.
So if the expected change is in minus, users should be informed to invest. The opposite, the users should be 
informed to withdraw money. 

{query}
'''

prompt = PromptTemplate(template = template, input_variables=['query'])

conv_chain = LLMChain(llm = conv_model, 
                      prompt = prompt,
                      verbose = True)

btc_price = df['Close'].iloc[0]
change = predictions[-1]
message = conv_chain.run(f'Bitcoin price - {btc_price}, expected change - {change}%')

start_index = message.find("We hope this email")

extracted_response = message[start_index:]

#========================EMAIL=================================

while True:
        print("===========================================")
        print("         Bitcoin price prediction")
        print("===========================================")
        print("1. Trenutna cena Bitcoin-a")
        print("2. Očekivana promena cene Bitcoin-a")
        print("3. Želim obaveštenje o promeni cene")
        print("0. Exit")
            
        answer = input("Unesite željenu opciju: ")

        if answer == "1":
            print(f"Trenutna cena Bitcoin-a je {df['Close'].iloc[0]}$")
        
        elif answer == "2":
            print(f"Očekivana promena cene Bitcoin-a je {predictions[0]}%")
        
        elif answer == "3":
            while True: 
                email = input("Unesite mail na koji želite da Vam stigne obaveštenje: ")
                if "@" in email:
                    sender = 'btcnotification036@gmail.com'
                    recivers = [email]
                    message = f"""From: BTC Notification from Gmail <btcnotification036@gmail.com>
                    To: {recivers[0]}
                    Subject: Bitcoin Notification

                    {extracted_response}


                    """
                    smtpObj = smtplib.SMTP('smtp.gmail.com', 587)
                    smtpObj.starttls() 
                    smtpObj.login(sender,'whdp xlyo zfux poxh')  

                    smtpObj.sendmail(sender, recivers, message)        
                    print("Uspešno je poslat mail.")

                    break
                
                else: 
                    wrong_input = input("Mail nije ispravan.\nŽelite li da ponovo pokušate: \n1 - da\n0 - ne\n")
                    if wrong_input != "1":
                        main()
                        break
        elif answer == "0":
            exit()
            break
            
        else:
            print("Izabrali ste nepostojeću opciju")


