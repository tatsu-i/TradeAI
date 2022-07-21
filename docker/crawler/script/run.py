# -*- coding: utf-8 -*-
import re
import os
import time
import sqlite3
from datetime import datetime
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from logging import getLogger, basicConfig, getLevelName, WARNING
from mackerel.client import Client
from module.symbol import TOPIX1000

logger = getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO")
basicConfig(level=getLevelName(log_level.upper()))

hub_host = os.environ.get("HUB_HOST", "hub")
db_path = os.environ.get("DB_PATH", "/data/emotion.db")

def mackerel_metric(metrics, service_name):
    try:
        client = Client(mackerel_api_key=os.environ.get("MACKEREL_API_KEY"))
        client.post_service_metrics(service_name, metrics)
    except Exception as e:
        logger.error(e)


def extract_nbr(input_str):
    if input_str is None or input_str == '':
        return 0
    try:
        return float(re.findall(r'-?\d+\.?\d*', input_str)[0])
    except Exception as e:
        print(e)
        print(input_str)
    return 0

def get_class(driver, class_type):
    try:
        return extract_nbr(driver.find_element_by_xpath(f'//*[@id="thread"]/div[3]/div/table/tbody/tr/td[@class="{class_type}"]').get_attribute('style'))
    except Exception as e:
        return 0

def get_emotion(symbol_name):
    result = {'strongest':0, 'strong': 0, 'both': 0, 'weak': 0, 'weakest': 0, 'total':0, 'error':0}
    emotion = {}
    options = webdriver.ChromeOptions()
    try:
        driver = webdriver.Remote(
            command_executor=f'http://{hub_host}:4444/wd/hub',
            desired_capabilities=DesiredCapabilities.CHROME)
        driver.get(f'https://stocks.finance.yahoo.co.jp/stocks/detail/?code={symbol_name}.T')
        driver.find_element_by_xpath('//*[@id="stockinf"]/ul/li[6]/a').click()
        emotion['strongest'] = get_class(driver, 'strongest')
        emotion['strong'] = get_class(driver, 'strong')
        emotion['both'] = get_class(driver, 'both')
        emotion['weak'] = get_class(driver, 'weak')
        emotion['weakest'] = get_class(driver, 'weakest')
        total = sum(emotion.values())
        emotion['total'] = total
        result = emotion
        driver.close()
        driver.quit()
        emotion['error'] = 0
    except:
        emotion['error'] = 1
    return result

if __name__ == '__main__':
    error = 0
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS emotion (key TEXT PRIMARY KEY, timestamp DATETIME, both INTEGER, strongest INTEGER, strong INTEGER, weak INTEGER, weakest INTEGER, total INTEGER, error INTEGER)")
    except sqlite3.Error as e:
        print('sqlite3.Error occurred:', e)


    for symbol_name in TOPIX1000:
        now = datetime.now()
        timestamp = str(datetime(now.year, now.month, now.day))
        key = f"{now.strftime('%Y%m%d')}_{symbol_name}"
        symbol_root = symbol_name.replace('-TS', '')
        emotion = get_emotion(symbol_root)
        emotion["key"] = key
        emotion["timestamp"] = timestamp
        if emotion["error"] != 0:
            error += 1

        try:
            cursor.execute(f"REPLACE INTO emotion VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (emotion["key"], emotion["timestamp"], emotion["both"], emotion["strongest"], emotion["strong"], emotion["weak"], emotion["weakest"], emotion["total"], emotion["error"]))
            connection.commit()
        except sqlite3.Error as e:
            print('sqlite3.Error occurred:', e)

        logger.info(f"get emotion {symbol_root} {emotion}")
        time.sleep(12)
        metrics = []
        metrics.append({
            'name': f'crawler.error',
            'time': time.time(),
            'value': error
        })
        mackerel_metric(metrics, "Stock-AI")
    connection.close()
