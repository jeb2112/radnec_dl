# start of attempt to parse tokens from html headers for use with cookies. switched to chromedriver instead
from html.parser import HTMLParser
import re
class ParseCode(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.recorddata = False
        self.code = None

    def handle_starttag(self,tag,attrs):
        if tag == 'script':
            self.recorddata=True

    def handle_data(self,data):
        if self.recorddata == True:
            print("Data: ",data)
            result = re.search('code=(.*?)\&session_state',data,re.UNICODE)
            if result:
                self.code = result.group(1)
            self.recorddata=False
