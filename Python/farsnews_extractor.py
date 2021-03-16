import HTMLParser



class NewsListExtractor(HTMLParser.HTMLParser):
    def __init__(self):
        HTMLParser.HTMLParser.__init__(self)
        self.news_list = []
        self.state = 'None'


    def handle_starttag(self, tag, attrs):
        if self.state == 'None' and tag == 'ul' and ('class', 'news-list list-unstyled p-0 m-0') in attrs:
            self.state = 'UL'
            return

        if self.state == 'UL' and tag == 'a' and ('class', 'col-4 px-0 mb-2 mb-sm-0') in attrs:
            for a in attrs:
                if a[0] == 'href':
                    self.news_list.append(a[1])


    def handle_endtag(self, tag):
        if tag == 'ul' and self.state == 'UL':
            self.state = 'None'



class NewsTextExtractor(HTMLParser.HTMLParser):
    def __init__(self):
        HTMLParser.HTMLParser.__init__(self)
        self.news_text = u''
        self.state = []


    def IsState(self, chk_state):
        return len(self.state) and self.state[-1] == chk_state


    def PushState(self, new_state):
        self.state.append(new_state)


    def PopState(self):
        if len(self.state):
            del self.state[-1]


    def handle_starttag(self, tag, attrs):
        if tag == 'div' and ('itemprop', 'articleBody') in attrs:
            self.PushState(tag)
        elif self.IsState('div') and tag == 'p':
            self.PushState(tag)


    def handle_endtag(self, tag):
        if self.IsState('p') and tag == 'p':
            self.PopState()
            self.news_text += u' '
        elif self.IsState('div') and tag == 'div':
            self.PopState()


    def handle_data(self, data):
        if self.IsState('div') or self.IsState('p'):
            self.news_text += data.decode('utf-8').strip()

