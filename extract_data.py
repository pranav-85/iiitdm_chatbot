from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
import time
import re
import os

def extract_curriculum(URL: str) -> None:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.page_load_strategy = 'eager' 

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(URL)

    wait = WebDriverWait(driver, 20)

    links = driver.execute_script("""
    return Array.from(document.querySelectorAll("a[href$='.pdf']")).map(a => a.href);
    """)

    driver.quit()

    print(links)

    for link in links:
        response = requests.get(link)
        DATA_DIR = '../data/'

        match = re.search(r"/([^/]+?\.pdf)(\?|#|$)", link)
        file_name = match.group(1)

        file_path = DATA_DIR  + file_name

        print(f'Downloading {file_name}...')

        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f'{file_name} downnloaded!')
        
    print(f'Successfully completed downloading!')
        

def main() -> None:
    URL = "https://www.iiitdm.ac.in/academics/study-at-iiitdm"
    extract_curriculum(URL)


if __name__ == '__main__':
    main()