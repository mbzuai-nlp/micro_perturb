import requests
import json
import asyncio
import os
import httpx
import time
import re
import argparse
from bs4 import BeautifulSoup
from paperdl import PaperClient
from pathlib import Path

ACL_BASE_URL = "https://aclanthology.org"
ARXIV_BASE_URL = "https://export.arxiv.org"

def process_paper_metadata(paper_element, section_element):
    a_ele = paper_element.select('span.d-block > strong > a')[0]
    authors_ele = paper_element.select('span.d-block > a')
    authors = []

    for author_ele in authors_ele:
        authors.append(author_ele.get_text(strip=True)) # probably no need for spacing here for child elements

    paper_id = a_ele['href']
    link = ACL_BASE_URL + a_ele['href']
    paper_title = a_ele.get_text("", strip=False)

    # get abstract
    try:
        abs_identifier = paper_element.select('div.d-block > span.d-inline > a.text-bg-info')[0]['aria-controls']
        abstract_element = section_element.find('div', id=abs_identifier)
        abs = abstract_element.select('div.card-body')[0].get_text(' ', strip=True)
    except:
        print(paper_title)
        abs = ""

    return {
        'id': paper_id,
        'url': link,
        'paper_title': paper_title,
        'authors': authors,
        'abstract': abs,
        'status': 'not_downloaded',
    }

def get_metadata(output_path, acl_url, main_count, findings_count):
    if os.path.exists(output_path):
        print(f"Metadata file {output_path} already exists. Skipping metadata collection.")
        return
    
    response = requests.get(acl_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    main_section = soup.find_all('div', id='2025emnlp-main')[0]
    findings_section = soup.find_all('div', id='2025findings-emnlp')[0]

    main_papers = main_section.find_all('div', class_='d-sm-flex')
    findings_papers = findings_section.find_all('div', class_='d-sm-flex')
    assert len(main_papers) == main_count
    assert len(findings_papers) == findings_count

    main_papers = main_papers[1:] #skip the first element which is the full proceedings
    findings_papers = findings_papers[1:]

    all_paper_metadata = []
    for paper_element in main_papers:
        paper_metadata = process_paper_metadata(paper_element, main_section)
        paper_metadata['venue'] = 'emnlp_2025_main'
        all_paper_metadata.append(paper_metadata)

    for paper_element in findings_papers:
        paper_metadata = process_paper_metadata(paper_element, findings_section)
        paper_metadata['venue'] = 'emnlp_2025_findings'
        all_paper_metadata.append(paper_metadata)

    with open(output_path, 'w') as f:
        json.dump(all_paper_metadata, f)

def save_paper_status(paper_status, status_file):
    with open(status_file, 'w') as f:
        json.dump(paper_status, f)
        
def load_paper_status(status_file):
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            paper_status = json.load(f)
    else:
        with open(metadata_file, 'r') as f:
            all_paper_metadata = json.load(f)

        paper_status = {}
        for paper_metadata in all_paper_metadata:
            paper_id = paper_metadata['id']
            paper_status[paper_id] = paper_metadata

        with open(status_file, 'w') as f:
            json.dump(paper_status, f)
    
    return paper_status

async def get_arxiv_data(metadata_file, output_status_file):
    paper_status = load_paper_status(output_status_file)
    
    timeout = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0)
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
    async with PaperClient(["arxiv"]) as client:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as req_client:
            for paper_id, paper_metadata in paper_status.items():
                paper_title = paper_metadata['paper_title']
                status = paper_metadata['status']

                # either downloaded or skipped
                if status != 'not_downloaded':
                    continue

                #ADD: multiple search results and fuzzy matching of abstract to find the best match
                paper_data = await client.search(paper_title, total_results=1)
                if len(paper_data) == 0:
                    print(f"Paper not found on arXiv: {paper_metadata['paper_title']}")
                    paper_metadata['status'] = 'skipped'
                    paper_metadata['skip_reason'] = 'not_found'
                    save_paper_status(paper_status, output_status_file)
                    continue

                arxiv_data = {
                    'arxiv_id': paper_data[0].arxiv_id,
                    'title': paper_data[0].title,
                    'abstract': paper_data[0].abstract,
                    'authors': paper_data[0].authors,
                    'article_url': paper_data[0].article_url,
                    'download_url': str(paper_data[0].download_url),
                    #'is_open_access': paper_data[0].is_open_access, DONT USE THIS, SOURCE CODE JUST SETS TO TRUE WITHOUT CHECKING
                }

                arxiv_url = ARXIV_BASE_URL + paper_data[0].article_url[len('https://arxiv.org'):]
                
                SLEEP_TIME = 3
                for i in range(5):
                    paper_page = await req_client.get(arxiv_url)
                    if len(paper_page.text) < 100 and i != 4:
                        #rate limit exceeded, wait and retry
                        print('Rate limit exceeded, waiting...')
                        time.sleep(SLEEP_TIME**(i+1))
                    elif len(paper_page.text) < 100 and i == 4:
                        print(f"Failed to fetch arXiv page for {paper_title} after 5 attempts.")
                        paper_metadata['status'] = 'skipped'
                        paper_metadata['skip_reason'] = 'failed_fetch'
                        save_paper_status(paper_status, output_status_file)
                        break

                if paper_metadata['status'] == 'skipped':
                    continue

                soup = BeautifulSoup(paper_page.text, 'html.parser')

                #license: link shows license type (CC type or arxiv nonexclusive license)
                license_element = soup.find('div', class_='abs-license')
                license_link = license_element.find('a')['href'] if license_element else None
                arxiv_data['license_url'] = license_link
                print(license_link)

                #download tex source
                access_ele = soup.find('div', class_='full-text')
                ul_ele = access_ele.find('ul') if access_ele else None
                tex_source_ele = ul_ele.find('a', string=re.compile(r'TeX Source')) if ul_ele else None
                tex_source_link = tex_source_ele['href'] if tex_source_ele else None
                tex_source_url = ARXIV_BASE_URL + tex_source_link if tex_source_link else None
                arxiv_data['tex_source_url'] = tex_source_url

                paper_metadata['status'] = 'metadata_downloaded'
                #paper_metadata['downloaded_path'] = str(paths[0])
                paper_metadata['arxiv_metadata'] = arxiv_data

                print(paper_metadata)
                with open(output_status_file, 'w') as f:
                    json.dump(paper_status, f)

    print('Metadata collection complete.')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Metadata Collector',
        description='Links manuscripts on an ACL page to ArXiV ids and metadata',
    )

    parser.add_argument('-e', '--event_name')
    parser.add_argument('-u', '--url')
    parser.add_argument('-mc', '--main_count', type=int)
    parser.add_argument('-fc', '--findings_count', type=int)

    args = parser.parse_args()

    event_name = args.event_name
    url = args.url
    main_count = args.main_count
    findings_count = args.findings_count

    #UNCOMMENT FOR NEW EVENT
    parent_directory = Path(f'./{event_name}')
    os.makedirs(parent_directory, exist_ok=True)
    metadata_file = parent_directory / f'acl_metadata.json'
    get_metadata(metadata_file, url, main_count, findings_count)
    output_status_file = parent_directory / f'arxiv_metadata.json'
    asyncio.run(get_arxiv_data(metadata_file, output_status_file))