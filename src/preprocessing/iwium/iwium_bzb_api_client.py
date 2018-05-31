import json
import os
import xml.etree.ElementTree as eT
from datetime import date, timedelta

import requests

BZP_API_URL = 'http://websrv.bzp.uzp.gov.pl/BZP_PublicWebService.asmx'
JSON_API_ENDPOINT = BZP_API_URL + '/ogloszeniaZP400KryteriaWyszukiwaniaJSON'

url = JSON_API_ENDPOINT
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'en-GB,en;q=0.9,pl;q=0.8',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Content-Length': '330',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Pragma': 'no-cache',
    'Referer': 'http://websrv.bzp.uzp.gov.pl/BZP_PublicWebService.asmx?op=ogloszeniaZP400KryteriaWyszukiwaniaJSON',
    'Upgrade-Insecure-Requests': '1'
}

data = {
    '_rodzaj_zamawiajacego': 99,
    '_rodzaj_zamowienia': 99,
    '_tryb_udzielenia_zamowienia': 99,
    '_numer_ogloszenia': '',
    '_data_publikacjiOd': '',
    '_data_publikacjiDo': '',
    '_nazwa_nadana_zamowieniu': '',
    '_cpv_glowny_przedmiot': '',
    '_czyGrupaCPV': 0,
    '_zamawiajacy_nazwa': '',
    '_zamawiajacy_miejscowosc': '',
    '_zamawiajacy_wojewodztwo': '',
    '_calkowita_wart_zam_od': -1,
    '_calkowita_wart_zam_do': -1
}


def fetch_data_for_day(curr_date):
    data['_data_publikacjiOd'] = str(curr_date)
    data['_data_publikacjiDo'] = str(curr_date)

    r = requests.post(url=url, data=data, headers=headers)
    response_xml = r.text

    tree = eT.fromstring(response_xml)
    string_tag_content = tree.text

    tender_list_json = json.loads(string_tag_content)

    return tender_list_json


def save_tender_data_for_found_bulletin_nums(single_day_tenders, bulletin_nums: list, json_save_dir):
    if len(single_day_tenders['Table']) == 0:
        print("No tenders found")
        return

    found_bulletin_nums = []

    for bulletin_num in bulletin_nums:
        found_tender = [tender for tender in single_day_tenders['Table'] if tender['biuletyn'] == bulletin_num]
        if len(found_tender) == 1:
            found_tender = found_tender[0]
            with open(json_save_dir + "/" + bulletin_num + ".json", 'w+') as json_tender_file:
                json.dump(found_tender, json_tender_file)
            found_bulletin_nums.append(bulletin_num)
    for found_bulletin_num in found_bulletin_nums:
        bulletin_nums.remove(found_bulletin_num)


def fetch_data_daily(bulletin_nums_of_tenders_to_find, json_save_dir: str):
    # start_date = date(2017, 1, 1)
    # end_date = date.today()
    start_date = date(2017, 6, 26)
    end_date = date(2017, 6, 26)

    curr_date = start_date

    total_days = (end_date - start_date).days + 1
    tenders_left = len(bulletin_nums_of_tenders_to_find['observed']) + len(
        bulletin_nums_of_tenders_to_find['reported']) + len(bulletin_nums_of_tenders_to_find['viewed'])
    print(tenders_left)

    counter = 0

    os.makedirs(json_save_dir + '/observed_json', exist_ok=True)
    os.makedirs(json_save_dir + '/viewed_json', exist_ok=True)
    os.makedirs(json_save_dir + '/reported_json', exist_ok=True)

    while curr_date <= end_date:
        single_day_tenders = fetch_data_for_day(curr_date)

        save_tender_data_for_found_bulletin_nums(single_day_tenders, bulletin_nums_of_tenders_to_find['observed'],
                                                 json_save_dir + '/observed_json')
        save_tender_data_for_found_bulletin_nums(single_day_tenders, bulletin_nums_of_tenders_to_find['reported'],
                                                 json_save_dir + '/reported_json')
        save_tender_data_for_found_bulletin_nums(single_day_tenders, bulletin_nums_of_tenders_to_find['viewed'],
                                                 json_save_dir + '/viewed_json')

        tenders_left = len(bulletin_nums_of_tenders_to_find['observed']) + len(
            bulletin_nums_of_tenders_to_find['reported']) + len(bulletin_nums_of_tenders_to_find['viewed'])

        print("{0} [{1:2.2d}]".format(curr_date, counter / total_days))
        print("tenders to find: {0}".format(tenders_left))

        curr_date = curr_date + timedelta(days=1)
        counter += 1
