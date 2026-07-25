from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.add_argument("--headless=new")
options.add_argument("--window-size=1280,800")

browser = webdriver.Chrome(options=options)

try:
    browser.get("https://ai-chatbot-fastapi-3b8w.onrender.com/")
    wait = WebDriverWait(browser, 10)
    h1 = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
    print({"title": browser.title, "h1": h1.text})
finally:
    browser.quit()