# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:05:29 2019

@author: jflygare
"""

from bs4 import BeautifulSoup
import requests
import os
from selenium import webdriver
import datetime
import pickle


os.chdir('C:/Users/jflygare/Documents/ML_Projects/Python/')

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

url ='https://bostad.stockholm.se/Lista/?s=52.88239&n=63.11464&w=3.18604&e=27.37793&sort=annonserad-fran-desc&ungdom=1'
driver.get(url) 

page_source = driver.page_source

master_soup = BeautifulSoup(page_source, 'lxml')

driver.quit()

baseUrl = 'https://bostad.stockholm.se'
s2 = 'Högsta tillåtna ålder: '

# Laddar in alla URL:er från senaste körningen
with open('bostad_scrape_results.pkl','rb') as f:  # Python 3: open(..., 'rb')
    apts_last_search = pickle.load(f)

my_date = datetime.datetime.strptime('2012-02-06','%Y-%m-%d')
apts = []
apts_new_search = []


for apartment in master_soup.find_all('div',class_='m-apartment-card'):
    try:
        apt_area = apartment.find('div',class_='m-apartment-card__area').text
        
        link_part = apartment.a['href']
        link = baseUrl+link_part
        apts_new_search.append(link)
        if link in apts_last_search:
            continue
        else:
            source = requests.get(link).content
            soup = BeautifulSoup(source,'lxml')
        
            article = soup.find('div',class_='article')
            s1 = article.text
            
            maxAge = s1[s1.index(s2) + len(s2):s1.index(s2) + len(s2) + 2]
            #minAge = s1[s1.index(s2b) + len(s2):s1.index(s2b) + len(s2b) + 2]
        
            if int(maxAge) >= 25:
                d = dict()
                
                rightPanel = soup.find('div',class_='col33')
                fakta = rightPanel.find('div',class_='inner').text
    
                hyra = fakta[fakta.index('kr/mån') - 5:fakta.index('kr/mån')+6]
                yta = fakta[fakta.index('kvm')-3:fakta.index('kvm')+3]
                kötid = rightPanel.find('div',id='statistik-box',class_='box secondary').text
    
                kötider = ','.join(kötid[kötid.index('2'):kötid.index('2')+32].split())
                
                top_queue = 4
                n_iter = 0
                for i in kötider.split(','):
                    n_iter = n_iter + 1
                    queue_date = datetime.datetime.strptime(i,'%Y-%m-%d')
                    if queue_date >= my_date:
                        top_queue = n_iter
                        break
                    else:
                        continue
                d['Area'] = apt_area
                d['Age'] = maxAge
                d['Rent'] = hyra
                d['Size'] = yta
                d['Queue'] = top_queue
                d['Link'] = link
                apts.append(d)
            else: 
                continue
    
    except:
        pass
    
#Skriver in senaste fångsten i pickle 
with open('bostad_scrape_results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(apts_new_search, f)


apts = sorted(apts,key=lambda i:(i["Queue"],i['Age']))

emailText = ''
for i in apts:
    emailText = emailText + str(i["Area"]) + str(i["Age"]) + " år" + ", " + "Köplats: " +  \
    str(i["Queue"]) + ", " + i["Rent"] + ", " + i["Size"] + "\n" + i["Link"] + "\n \n"
    
    
print(emailText)
## Skickar mailet
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
    
EMAIL_ADDRESS = os.environ.get('EMAIL_ACC_ML')
EMAIL_PASSWORD = os.environ.get('EMAIL_PW_ML')

recipients = ['flygandejakob16@gmail.com','paumen15@gmail.com']
msg = MIMEMultipart()
msg['From'] = EMAIL_ADDRESS
msg['To'] = ', '.join(recipients)
    
if len(apts) > 0:

    msg['Subject'] = 'NEW APARTMENTS!!'
    msg.attach(MIMEText(emailText))
    
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)
else:
    msg['Subject'] = 'Sad news: Nothing new :('
    msg.attach(MIMEText('Hopefully tomorrow brings better prospects'))
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)