import collections
import json
import xml.etree.ElementTree as ET
from typing import Type, List

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
    '_numer_ogloszenia': '608696',
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


def fetch_and_save_tender_json(valid_bulletin_number: str, json_save_dir: str):
    """
    :param valid_bulletin_number: <int>-N-<year>
    :param json_save_dir:
    :return:
    """
    data['_numer_ogloszenia'] = valid_bulletin_number
    r = requests.post(url=url, data=data, headers=headers)
    response_xml = r.text

    tree = ET.fromstring(response_xml)
    string_tag_content = tree.text

    tender_list_json = json.loads(string_tag_content)

    if len(tender_list_json['Table']) != 1:
        print('Number of tenders len different than 1')
    elif len(tender_list_json['Table']) > 0:
        with open(json_save_dir + "/" + valid_bulletin_number + ".json", 'w+') as json_tender_file:
            json.dump(tender_list_json['Table'][0], json_tender_file)
    else:
        print('no tenders for given id: ' + valid_bulletin_number)


def save_jsons(tender_numbers: List[str], sub_dir_name: str):
    for tender_number in tender_numbers:
        fetch_and_save_tender_json(tender_number, sub_dir_name)
