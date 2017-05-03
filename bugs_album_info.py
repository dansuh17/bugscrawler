from bs4 import BeautifulSoup
import os
import requests
import pandas as pd


if __name__ == '__main__':
    album_dir = 'bugs_albums'
    album_ids = os.listdir(album_dir)
    albumid = album_ids[1]
    to_replace_char = ['\t', '\n', '\r']

    series_dict = {}
    for albumid in album_ids:
        album_url = 'http://music.bugs.co.kr/album/{}'.format(albumid)
        page = requests.get(album_url)
        soup = BeautifulSoup(page.text, 'html.parser')

        # retrieve album title
        title_box = soup.find('header', class_='pgTitle')
        album_title = title_box.div.h1.string.strip()
        print('Processing: {} - {}'.format(albumid, album_title))
        if 'title' not in series_dict:
            series_dict['title'] = {}
        series_dict['title'][albumid] = album_title

        # retrieve all album attributes
        found_table = soup.find('table', class_='info')
        try:
            for tr in found_table.tbody.find_all('tr'):
                field_name = tr.th.string
                field_value = tr.td.get_text().strip()

                # replace tabs, carriage returns, returns, etc.
                for char in to_replace_char:
                    field_value = field_value.replace(char, '')

                # create a dict if the field name doesn't exist
                if field_name not in series_dict:
                    series_dict[field_name] = {}

                series_dict[field_name][albumid] = field_value
        except AttributeError:
            print('No sufficient atributes for id : {}'.format(albumid))
            continue

        # find the image
        img_dir = 'album_images'
        information_box = soup.find('div', class_='basicInfo')
        album_img = information_box.find('img')

        try:
            img_url = album_img.attrs['src']
            with open(os.path.join(img_dir, '{}.jpg').format(albumid), 'wb') as img_file:
                res = requests.get(img_url, stream=True)
                if not res.ok:
                    print(res)
                for block in res.iter_content(1024):
                    if not block:
                        break
                    img_file.write(block)
        except AttributeError:
            print('No image for id : {}'.format(albumid))
            continue

    # convert to pandas Series for convenient transformation into DataFrame
    for key in series_dict:
        series_dict[key] = pd.Series(series_dict[key])
    album_info_df = pd.DataFrame(series_dict)

    # finally - write to csv
    album_info_df.to_csv('album_info.csv', encoding='utf-8', header=True, index=True)
