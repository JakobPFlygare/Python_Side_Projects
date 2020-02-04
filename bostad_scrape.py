# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:05:29 2019

@author: jflygare
"""

from bs4 import BeautifulSoup
import requests
#import csv
import datetime
import os
#import pandas as pd
from selenium import webdriver
import datetime

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
#s2b = 'Lägsta tillåtna ålder: '

#csv_file = open('bostadScrape.csv','w',encoding="utf-8")
#csv_writer = csv.writer(csv_file)  
#csv_writer.writerow(['link','rent','wait_times','fetch_date'])
#csv_writer.writerow(['link'])
#csv_file.close()


#df  = pd.read_csv('bostadScrape.csv',sep=",")
all_links = ''
my_date = datetime.datetime.strptime('2012-02-06','%Y-%m-%d')
apts = []
#d = dict()
#apartment = master_soup.find('div',class_='m-apartment-card')
for apartment in master_soup.find_all('div',class_='m-apartment-card'):
    try:
        link_part = apartment.a['href']
        link = baseUrl+link_part
        
        apt_area = apartment.find('div',class_='m-apartment-card__area').text
        
        source = requests.get(link).content
        soup = BeautifulSoup(source,'lxml')
    
        article = soup.find('div',class_='article')
        s1 = article.text
        
        maxAge = s1[s1.index(s2) + len(s2):s1.index(s2) + len(s2) + 2]
        #minAge = s1[s1.index(s2b) + len(s2):s1.index(s2b) + len(s2b) + 2]
    
        if int(maxAge)>=25:
            d = dict()
            
            #print(link)
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
                
            #csv_writer.writerow([link,hyra,kötider,datetime.datetime.today()])
            #csv_writer.writerow([link])
            d['Area'] = apt_area
            d['Age'] = maxAge
            d['Rent'] = hyra
            d['Size'] = yta
            d['Queue'] = top_queue
            d['Link'] = link
            apts.append(d)
            
            
            #all_links = all_links + '\n' + str(maxAge) + '\n' + '[' + kötider + ']' + '\n' + link + '\n'
        else: 
            continue

    except:
        pass
    
    
print(all_links)

apts_sort = sorted(apts,key=lambda i:(i["Queue"],i['Age']))
emailText = ''
for i in apts_sort:
    emailText = emailText + str(i["Area"]) + "Age:" + str(i["Age"]) + ", " + "Top " +  \
    str(i["Queue"]) + ", " + i["Rent"] + ", " + i["Size"] + "\n" + i["Link"] + "\n \n"
    
    
print(emailText)
    
#csv_file.close()


## Skickar mailet

import smtplib
from email.message import EmailMessage
from email.mime.text import MIMEText

EMAIL_ADDRESS = 'jakobflygareml2@gmail.com'
EMAIL_PASSWORD = 'brandbil123'

    
msg = EmailMessage()
msg['Subject'] = 'UNGDOMSLÄGENHETER!!'
msg['From'] = EMAIL_ADDRESS
msg['To'] = 'flygandejakob16@gmail.com'


msg.set_content(emailText)

with open('bostadScrape.csv') as fp:
    # Create a text/plain message
    msg.set_content(MIMEText(fp.read()))


with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    smtp.send_message(msg)
