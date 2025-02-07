import requests
from bs4 import BeautifulSoup
import pandas as pd


# Hàm để lấy nội dung của bài báo từ URL
def get_article_content(article_url):
    try:
        response = requests.get(article_url)
        html_content = response.content
        soup = BeautifulSoup(html_content, "html.parser")

        paragraphs = soup.find_all("p", class_="Normal")
        filtered_paragraphs = [p for p in paragraphs if not p.find("strong")]
        article_text = " ".join([p.get_text().strip() for p in filtered_paragraphs])
        article_text = article_text.replace("\n", "").replace("\r", "").strip()

        return article_text
    except Exception as e:
        print("Error:", e)
        return None


# Hàm để lấy các href từ một trang và trả về danh sách
def get_href_from_page(url):
    href_list = []
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, "html.parser")
    h2_tags = soup.find_all("h2", class_="title-news")
    for h2_tag in h2_tags:
        a_tag = h2_tag.find("a")
        if a_tag:
            href = a_tag.get("href")
            if href and href not in href_list:
                href_list.append(href)
    return href_list


# Hàm để lấy nội dung từ nhiều trang và lưu vào file Excel
def get_articles_from_multiple_pages(base_url, num_pages):
    articles = []
    for page in range(1, num_pages + 1):
        url = f"{base_url}-p{page}"
        href_list = get_href_from_page(url)
        for href in href_list:
            article_content = get_article_content(href)
            if article_content:
                articles.append(article_content)
    return articles


# Thay link cua chu de can lay vao day
base_url = "https://vnexpress.net/the-thao"

# Số trang muốn lấy dữ liệu
num_pages = 15

# Lấy nội dung từ nhiều trafvdng
articles = get_articles_from_multiple_pages(base_url, num_pages)

# Tạo DataFrame từ danh sách các bài báo
df = pd.DataFrame({"Content": articles})

# Lưu DataFrame vào file Excel
df.to_excel("the_thao.xlsx", index=False)

print("Dữ liệu đã được lưu vào file Excel.")
