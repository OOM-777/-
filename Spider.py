import os
import requests
from bs4 import BeautifulSoup


class Spider:
    # 构造函数
    def __init__(self, params=None):
        self.count = 0
        self.inbox_url = "http://www.yopmail.com/zh/inbox.php"
        self.mail_url_head = 'http://www.yopmail.com/zh/'
        # 请求头
        self.headers = headers = {
            'Cookie': 'colaw=0; tc=1; localtime=18:35; params=0.0; cnl=1; __gads=ID=83d9c3cb1bea8e09-22a9efc860c5006d:T=1609324332:RT=1609324332:S=ALNI_MaAH85ZP69Iq8-3degtfM5YMzzqRA; _ga=GA1.2.923250923.1609324331; _gid=GA1.2.1678912217.1609324333; compte=booooi; PHPSESSID=51l0md7794b4mp125ppetu1532; ys=CZmL2ZmxlZwVkZQx3ZGD5AGD; yc=FZwN3ZGH2ZwD2BGxkBGD0AGR',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
        }
        # 判断是否在实例化时给params参数
        if params is not None:
            self.params = params
        else:
            # 默认URL请求参数
            self.params = {
                "login": "booooi",
                "p": "1",
                "yp": "AAQR3BQHmAwN2ZwV4AQN3AGp",
                "yj": "WZGpkAmD0BGpmAwt5BQL2ZQH",
                "v": "3.1"
            }

    # 获取收件箱HTML
    def get_inbox(self):
        response = requests.get(self.inbox_url, params=self.params)
        print(response.url) # TEST
        return response.text

    # 将邮件内容保存到文件
    def save2file(self, text):
        if not os.path.exists('./data/data/data_get'):
            os.mkdir('./data/data/data_get')
        with open(f'./data/data/data_get/{str(self.count).zfill(3)}', 'w', encoding='UTF-8') as f:
            f.write(text)

    # 解析邮件内容
    def get_mails(self,n):
        inbox = self.get_inbox()
        # 解析每封邮件URL
        soup = BeautifulSoup(inbox, 'html.parser')
        # 查找所有名为m的div标签，返回类型为列表
        items = soup.findAll('div', {'class': 'm'})
        count = 0
        # 遍历列表
        for itm in items:
            if n==self.count:
                return
            # 在每个div标签查找href属性，即邮件对应链接
            temp = itm.find('a').get('href')
            # 合并链接(完整邮件链接)
            # 示例：http://www.yopmail.com/zh/m.php?b=booooi&id=me_ZwNkZwZjZQp1AmV0ZQNjZGp3BQx4AN==
            temp = self.mail_url_head + temp
            # 对链接发送请求，获取邮件内容
            email = requests.get(temp, headers=self.headers)
            # 改变网页编码为UTF-8
            email.encoding = 'utf-8'
            # 解析邮件
            email_contant = BeautifulSoup(email.text, 'html.parser')
            # 获取邮件内容
            email_contant = email_contant.find(
                'div', {'id': 'mailmillieu'}).text
            # 去除多余换行和制表符
            email_contant = email_contant.replace(
                '\n', ' ').replace('\t', ' ').strip()
            print(email_contant)  # TEST
            # 若邮件非空(邮件中图片，文件无法解析)则保存到文件中
            if email_contant:
                self.save2file(email_contant)
            else:
                continue
            self.count += 1


if __name__ == "__main__":
    # 实例化Spider对象
    spider = Spider()
    # 获取邮件
    spider.get_mails(n=9)
