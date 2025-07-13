import requests
from bs4 import BeautifulSoup

def crawl(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return ""

def main():
    text = crawl("https://www.gutenberg.org/files/1342/1342-h/1342-h.htm")
    with open("crawled_text.txt", "w") as f:
        f.write(text)
    print("Crawling complete.")

if __name__ == "__main__":
    main()
