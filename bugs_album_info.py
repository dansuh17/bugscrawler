from bs4 import BeautifulSoup
import os
import requests
import threading
import queue
import pandas as pd


class AlbumInfo:
    def __init__(self):
        super().__init__()
        self.series_dict = {}
        self.to_replace_char = ['\t', '\n', '\r']
        self.album_url_base = 'http://music.bugs.co.kr/album/{}'
        self.img_dir = 'album_images'

    def save_image(self, albumid):
        album_url = self.album_url_base.format(albumid)
        page = requests.get(album_url)
        soup = BeautifulSoup(page.text, 'html.parser')

        try:
            information_box = soup.find('div', class_='basicInfo')
            album_img = information_box.find('img')
            img_url = album_img.attrs['src']
            with open(os.path.join(self.img_dir, '{}.jpg').format(albumid),
                      'wb') as img_file:
                res = requests.get(img_url, stream=True)
                if not res.ok:
                    print(res)
                for block in res.iter_content(1024):
                    if not block:
                        break
                    img_file.write(block)
        except AttributeError:
            print('No image for id : {}'.format(albumid))
            return

    def save_info(self, albumid):
        album_url = self.album_url_base.format(albumid)
        page = requests.get(album_url)
        soup = BeautifulSoup(page.text, 'html.parser')

        # retrieve album title
        try:
            title_box = soup.find('header', class_='pgTitle')
            album_title = title_box.div.h1.string.strip()
            print('Processing: {} - {}'.format(albumid, album_title))
            if 'title' not in self.series_dict:
                self.series_dict['title'] = {}
            self.series_dict['title'][albumid] = album_title
        except AttributeError:
            return

        # retrieve all album attributes
        found_table = soup.find('table', class_='info')
        try:
            for tr in found_table.tbody.find_all('tr'):
                field_name = tr.th.string
                field_value = tr.td.get_text().strip()

                # replace tabs, carriage returns, returns, etc.
                for char in self.to_replace_char:
                    field_value = field_value.replace(char, '')

                # create a dict if the field name doesn't exist
                if field_name not in self.series_dict:
                    self.series_dict[field_name] = {}

                self.series_dict[field_name][albumid] = field_value
        except AttributeError:
            print('No sufficient atributes for id : {}'.format(albumid))
            return

    def create_csv(self):
        # convert to pandas Series for convenient transformation into DataFrame
        for key in self.series_dict:
            self.series_dict[key] = pd.Series(self.series_dict[key])
        album_info_df = pd.DataFrame(self.series_dict)

        # finally - write to csv
        album_info_df.to_csv('album_info.csv', encoding='utf-8',
                             header=True, index=True)


class WorkerThread(threading.Thread):
    def __init__(self, album_info_inst, queue):
        super().__init__()
        self.ai = album_info_inst
        self.queue = queue

    def run(self):
        while True:
            albumid_message = self.queue.get()
            if isinstance(albumid_message, str) and albumid_message == 'quit':
                break
            ai.save_image(albumid_message)
            ai.save_info(albumid_message)


def build_worker_pool(queue, size, album_info):
    workers = []
    for _ in range(size):
        worker = WorkerThread(album_info, queue)
        worker.start()
        workers.append(worker)
    return workers


if __name__ == '__main__':
    album_dir = 'bugs_albums'
    album_ids = os.listdir(album_dir)

    ai = AlbumInfo()
    thread_info_list = []
    queue = queue.Queue()
    worker_threads = build_worker_pool(queue, 4, ai)

    for albumid in album_ids:
        queue.put(albumid)

    for worker_t in worker_threads:
        queue.put('quit')

    for worker_t in worker_threads:
        worker_t.join()

    ai.create_csv()
