import yagmail
import os
import datetime as dt
import dotenv
dotenv.load_dotenv()

class MailServer():
    def send(title, contents):
        yag = yagmail.SMTP(os.getenv('MAIL_NAME'), os.getenv('MAIL_TOKEN'))

        contents = [
            '<html>'
            '<h1>北京时间</h1>',
            f'<h2>{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h2>',
            html_table,
            '</html>',
        ]
        yag.send('88therisingsun@gmail.com',monitor.title, contents=contents)

# Alternatively, with a simple one-liner:
# yagmail.SMTP('mygmailusername').send('to@someone.com', 'subject', contents)