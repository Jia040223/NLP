import requests
from bs4 import BeautifulSoup
from baseset import *
import datetime


class AuthorSpider():
    def __init__(self, header=None):
        self.delta = datetime.timedelta(days=1)
        #self.startday = datetime.date.today() - self.delta
        self.startday = datetime.date(2021, 7, 19)
        self.terminalday = datetime.date(2021, 1, 1)
        self.parseday = self.startday
        self.base_urls = 'http://paper.people.com.cn/rmrb/html/'
        self.timeline = (self.startday - self.terminalday).days

        if header == None:
            self.headers = {"Accept-Encoding": "gzip, deflate",
                            "Accept-Language": "zh-CN,zh;q=0.9",
                            "Host": "paper.people.com.cn",
                            "Proxy-Connection": "keep-alive"
                            }
            self.headers['User-Agent'] = user_agent()
        else:
            self.headers = header

        self.proxies = {}
        self.proxies['http'] = r_proxies()

        self.total_web_page = 0


    def get_url_text(self, url):
        text = ""
        try:
            resp = requests.get(url=url, headers=self.headers)
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, 'html.parser')
        except:
            print("爬取新闻内容失败：url不正确")
            return text

        if resp.status_code != 200:
            print("爬取失败内容失败：request返回不正常")
            return text

        title = soup.title.text.strip()  ##获得标题
        text += title

        div_tags = soup.find_all('div', class_="article")
        for div_tag in div_tags:
            try:
                for p_tag in div_tag.find_all("p"):
                    text += p_tag.text.strip()  ##得到文本, strip()去除空格
            except:
                continue

        return text


    def crawl_all(self):
        if self.timeline >= 0:
            page_links = self.base_urls + self.parseday.isoformat()[:-3] + "/" + self.parseday.isoformat()[8:10] + "/" + "nbs.D110000renmrb_"
            for i in range(1, 21):
                if i < 10:
                    url = page_links + "0" + str(i) + ".htm"
                else:
                    url = page_links + str(i) + ".htm"
                print("正在爬取新闻主页:", url)
                try:
                    url_list = []  ##空列表
                    resp = requests.get(url=url, headers=self.headers)
                    resp.encoding = resp.apparent_encoding
                    soup = BeautifulSoup(resp.text, 'html.parser')
                except:
                    print("爬取新闻主页失败：url不正确")
                    continue

                if resp.status_code != 200:
                    print("爬取新闻主页失败：request返回不正常", resp.status_code)
                    continue

                div_tags = soup.find_all('div', class_="news")

                for div_tag in div_tags:
                    try:
                        for a_tag in div_tag.find_all("a"):
                            prefix_url = self.base_urls + self.parseday.isoformat()[:-3] + "/" + self.parseday.isoformat()[8:10] + "/"
                            url_list.append((str(prefix_url + a_tag.get('href')).strip()))  ##得到href
                    except:
                        continue

                url_list = list(set(url_list))  ##去重

                with open('ch_content.txt', 'a', encoding='utf-8') as f:
                    for news_url in url_list:
                        news = self.get_url_text(news_url)
                        if news != "":
                            f.write(news)
                            self.total_web_page += 1


            self.parseday -= self.delta
            self.timeline -= 1
            self.crawl_all()


if __name__ == '__main__':
    myspider = AuthorSpider()
    myspider.crawl_all()
    print("总共爬取文章:", myspider.total_web_page)
