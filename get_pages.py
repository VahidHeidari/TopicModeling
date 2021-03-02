import codecs
import os
import sys
import HTMLParser

import requests

import farsnews_extractor
import remove_list



FARSNEWS_BASE_URL    = 'https://www.farsnews.ir/'
FARSNEWS_ECONOMY_FMT = FARSNEWS_BASE_URL + 'economy/macroeconomics?p={}'
FARSNEWS_WORLD_FMT   = FARSNEWS_BASE_URL + 'world/Analysis-International?p={}'

NUM_LIST_PAGES = 56

IS_DUMP_RAW_TEXT = False



def GetPage(page_path, url):
    # Check already fetched or not.
    if os.path.isfile(page_path):
        with open(page_path, 'rb') as f:
            return f.read()

    # Get not existing page.
    print('page `' + page_path + '\' not exists!')
    print('Get Page URL : ' + url)
    resp = requests.get(url)
    if resp.status_code != 200:
        print('{} Could not fetch page!'.format(resp.status_code))
        return ''

    # Write contents to file.
    print('Write page ' + os.path.basename(page_path) + ' to `' + page_path + '\' file!')
    base_dir = os.path.dirname(page_path)
    if len(base_dir) and not os.path.isdir(base_dir):
        print('Creating `' + base_dir + '\' directory!')
        os.makedirs(base_dir)
    with open(page_path, 'wb') as f:
        f.write(resp.content)
    return resp.content



def GetNewsLists(base_dir, url_fmt):
    news_list = set()
    for i in range(1, NUM_LIST_PAGES + 1):
        page_path = os.path.join('Datasets', base_dir, 'news_list', '{}.html'.format(i))
        url = url_fmt.format(i)
        page = GetPage(page_path, url)
        extr = farsnews_extractor.NewsListExtractor()
        extr.feed(page)
        news_list |= set([os.path.dirname(u) for u in extr.news_list])
    print('Num URLs in `{}\' : {}'.format(base_dir, len(news_list)))
    return list(news_list)



def CountSorter(a, b):
    if a[1] < b[1]:
        return 1
    if a[1] > b[1]:
        return -1
    if a[0] < b[0]:
        return 1
    if a[0] > b[0]:
        return -1
    return 0



def GetNewsPages(base_dir, news_list):
    for i in range(len(news_list)):
        l = news_list[i]

        # Check extracted text file.
        base_txt_name = '{}.txt'.format(os.path.basename(l))
        text_path = os.path.join('Datasets', base_dir, 'texts', base_txt_name)
        if os.path.isfile(text_path):
            continue

        # Extract news page texts.
        base_html_name = '{}.html'.format(os.path.basename(l))
        page_path = os.path.join('Datasets', base_dir, 'pages', base_html_name)
        url = FARSNEWS_BASE_URL[:-1] + l
        page_content = GetPage(page_path, url)

        print('extracting words from `' + page_path + '\'  #{} of {} . .  .'.format(i, len(news_list)))
        extr = farsnews_extractor.NewsTextExtractor()
        extr.feed(page_content)

        # Filter unwanted strings and characters.
        news_text = extr.news_text
        end_of_news = u'\u0646\u062a\u0647\u0627\u06cc \u067e\u06cc\u0627\u0645/\u0645'
        if news_text.endswith(end_of_news):
            news_text = news_text[:-len(end_of_news) - 1]
        for w in remove_list.REMOVE_LIST:
            news_text = news_text.replace(w, u' ')

        # Remove double spaces.
        rpl_text = news_text.replace(u'  ', u' ')
        while len(rpl_text) != len(news_text):
            news_text = rpl_text
            rpl_text = news_text.replace(u'  ', u' ')

        # Split words.
        words = news_text.split()

        # Read stop words.
        stop_words_path = os.path.join('Datasets', 'stop_words.txt')
        STOP_WORDS = [l.strip() for l in codecs.open(stop_words_path, 'r', encoding='utf-8')]

        # Count words.
        words_dict = {}
        for w in words:
            w = w.strip()
            if w in STOP_WORDS:
                continue

            if len(w) and words_dict.has_key(w):
                words_dict[w] += 1
            else:
                words_dict[w] = 1

        # Write results to text file.
        print('Write news text to file -> ' + text_path + '.txt')
        if not os.path.isdir(os.path.dirname(text_path)):
            os.makedirs(os.path.dirname(text_path))

        if IS_DUMP_RAW_TEXT:
            # Dump raw text.
            with open(text_path, 'w') as f:
                f.write(news_text.encode('utf-8'))
        else:
            # Dump word counts.
            sorted_words = sorted([(k, words_dict[k]) for k in words_dict], cmp=CountSorter)
            with open(text_path, 'wb') as f:
                for w in sorted_words:
                    s = u'{:3d}\t'.format(w[1]) + w[0] + u'\r\n'
                    f.write(s.encode('utf-8'))



if __name__ == '__main__':
    word_news_list = GetNewsLists('world', FARSNEWS_WORLD_FMT)
    GetNewsPages('world', word_news_list)
    economy_news_list = GetNewsLists('economy', FARSNEWS_ECONOMY_FMT)
    GetNewsPages('economy', economy_news_list)

