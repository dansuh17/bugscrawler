from bs4 import BeautifulSoup
import os
import requests
import sys
import time
from requests.exceptions import ConnectionError


class AlbumDescFileExistsException(Exception):
    def __init__(self, albumid):
        self.albumid = albumid


def is_noisy_line(line, target_noise_words):
    for noise_word in target_noise_words:
        if noise_word in line:
            return True
    return False


def store_album_desc(file_save_path, albumid):
    # words that are considered 'noise' to corpus
    target_noise_words = ['credit', '크래딧', '크레딧']

    album_url = 'http://music.bugs.co.kr/album/{}'.format(albumid)
    album_page = requests.get(album_url)
    soup = BeautifulSoup(album_page.text, 'html.parser')
    album_description_tag = str(soup.find(id='albumContents'))
    soup = BeautifulSoup(album_description_tag)

    # the actual text split by newline
    album_desc_text = soup.getText('\n').split('\n')

    if (album_desc_text is not None) and ('None' not in album_desc_text):
        filepath = os.path.join(file_save_path, str(albumid))
        if os.path.exists(filepath):
            raise AlbumDescFileExistsException(albumid)
        with open(filepath, 'w') as fi:
            for line in album_desc_text:
                if not is_noisy_line(line.lower(), target_noise_words):
                    fi.write(line)
                    print(line)


def crawl_new_albums():
    page_num = 1
    file_save_path = 'bugs_albums'
    if not os.path.exists(file_save_path):
        os.mkdir(file_save_path)

    while True:
        url = 'http://music.bugs.co.kr/newest/album/total?page={}'.format(page_num)
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')

        # get the list of albums
        albumlist = soup.find_all('ul', class_='albumList')
        list_soup = BeautifulSoup(str(albumlist[0]), 'html.parser')
        figures = list_soup.find_all('figure', class_='albumInfo')

        for figure_tag in figures:
            albumid = figure_tag['albumid']  # retrieve the album's id
            try:
                store_album_desc(file_save_path, albumid=albumid)
            except AlbumDescFileExistsException as adfee:
                print('The file for {} already exists. Breaking loop.'
                      .format(adfee.albumid))

        # search for next page
        page_num += 1


def crawl_by_albumid(starting_albumid, max_album_id):
    file_save_path = 'bugs_albums'

    while starting_albumid < max_album_id:
        try:
            print('ALBUM_ID : {}'.format(starting_albumid))
            store_album_desc(file_save_path, starting_albumid)
            starting_albumid += 1
        except ConnectionError:
            time.sleep(20)
        except KeyboardInterrupt:
            print('\nBye')
            sys.exit(0)


if __name__ == '__main__':
    # crawl_by_albumid(229631, 40000000)
    crawl_new_albums()
