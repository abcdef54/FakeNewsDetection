import requests
from typing import Dict, Any, List
import json
from bs4 import BeautifulSoup
import os
import re

class Scrappers:
   session = requests.Session()
   session.headers.update({
   "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) Gecko/20100101 Firefox/139.0",
   "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
   "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"
   })
   session.verify = False  # Disable SSL verification globally for this session


   SITE_CONFIG = {
      'vnexpress': {
         'source': 'vnexpress',
         'paragraph_selector': 'article.fck_detail',
         'date_published_prop': [('itemprop', 'datePublished')],
         'date_modified_prop': [('itemprop', 'dateModified')],
         'author_prop': [('name', 'authorInfo'), ('name', 'author'), ('meta', 'author-name')] # last element is the tag and the value to take
      },
      'tuoitre': {
         'source': 'tuoitre',
         'paragraph_selector': 'div[itemprop="articleBody"]',
         'date_published_prop': [('property', 'article:published_time')],
         'date_modified_prop': [('property', 'article:modified_time')],
         'author_prop': [('property', 'dable:author')]
      },
      'thanhnien': {
         'source': 'thanhnien',
         'paragraph_selector': 'div[itemprop="articleBody"]',
         'date_published_prop': [('property', 'article:published_time')],
         'date_modified_prop': [('itemprop', 'dateModified')],
         'author_prop': [('property', 'dable:author')]
      },
      'kenh14': {
         'source': 'kenh14',
         'paragraph_selector': 'div.detail-content.afcbc-body',
         'date_published_prop': [('property', 'article:published_time')],
         'date_modified_prop': [('name', 'hideLastModifiedDate')],
         'author_prop': [('property', 'article:author')]
      },
      'soha': {
         'source': 'soha',
         'paragraph_selector': 'div.detail-content.afcbc-body',
         'date_published_prop': [('property', 'article:published_time')],
         'date_modified_prop': [('name', 'hidLastModifiedDate')],
         'author_prop': [('property', 'article:author')]
      },
      'gamek': {
         'source': 'gamek',
         'paragraph_selector': 'div.rightdetail_content.detailsmallcontent',
         'date_published_prop': [('property', 'article:published_time')],
         'date_modified_prop': [('property', 'article:modified_time')],
         'author_prop': [('name', 'author')]
      },
      'theanh28': {
         'source': 'theanh28',
         'paragraph_selector': 'div.message-content.js-messageContent article.message-body',
         'date_published_prop': [('json-ld', 'datePublished')],
         'date_modified_prop': [('json-ld', 'dateModified')],
         'author_prop': [('class', 'message--post')],
      }
   }



   def __init__(self) -> None:
      self.bs = None
      self.type: str = ""
      self.result: Dict[str, Any] = {}


   def _extract_tag(self, sel_key_pairs: List[Tuple[str, str]], tags: List[str] = ["meta"],
                    default_value=None, attr: str = 'content') -> str | None:
      """
      Try multiple tags and selector, key pairs to find the first matching element.
       
      Args:
         tags (List[str]): List of tag names to search for.
         sel_key_pairs (List[Tuple[str, str]]): List of tuples containing selector and key pairs.
         default_value (str, optional): Default value to return if no matching element is found.
         attr (str): The attribute to extract from the found element.
      """

      if not isinstance(tags, list):
         tags = [tags]

      for tag in tags:
         for sel, key, in sel_key_pairs:
            element = self.bs.find(tag, {sel: key}) # type:ignore
            if element:
                  return element.get(attr, default_value) if default_value is not None else element.get(attr) # type:ignore
      return default_value

   
   def _extract_paragraphs(self, selector: str) -> str:
      paragraph_tags = self.bs.select_one(selector) # type:ignore
      if not paragraph_tags:
         raise ValueError("No paragraph tags found in the HTML content.")
      
      if self.type == 'theanh28':
         br_tags = paragraph_tags.find_all('br')
         for br in br_tags:
            br.replace_with('\n')
         return self._clean_text(paragraph_tags.get_text(strip=True, separator=' ')) if paragraph_tags else 'No paragraphs found' # type:ignore
      else:
         p_tags = paragraph_tags.find_all('p') 

      if not p_tags:
         raise ValueError("No paragraph tags found in the specified selector.")
      
      return self._clean_text('\n'.join([p.get_text(strip=True, separator=' ') for p in p_tags])) if p_tags else 'No paragraphs found' # type:ignore
   

   def scrape(self, web_key: str) -> Dict[str, Any]:
      config: Dict[str, Any] = self.SITE_CONFIG.get(web_key) # type:ignore
      if not config:
         raise ValueError(f"Configuration for {web_key} not found in SITE_CONFIG.")

      source = config['source']
      title = self._extract_tag([("property", "og:title")])
      url = self._extract_tag([("property", "og:url")])
      image = self._extract_tag([("property", "og:image")])
      description = self._extract_tag([("property", "og:description")])
      copyright = self._extract_tag([("name", "copyright")])
      language = self._extract_tag([("property", "og:locale"), ("name", "language"), ("itemprop", "inLanguage")], default_value='vi')

      if web_key == 'theanh28':
         date_published = self._extract_json_ld_data('datePublished')
      else:
          date_published = self._extract_tag(config['date_published_prop'])

      if web_key in ['kenh14', 'soha']:
         date_modified = self._extract_tag(config['date_modified_prop'], tags=['input'], attr='value')
      elif web_key == 'theanh28':
         date_modified = self._extract_json_ld_data('dateModified')
      else:
         date_modified = self._extract_tag(config['date_modified_prop'])

      if web_key == 'vnexpress':
         # special handling for vnexpress because they suck
         author = self._extract_tag(config['author_prop'], attr='author-name')  or\
         self._extract_tag(config['author_prop'])
      elif web_key == 'theanh28':
         # special handling for theanh28 because they suck
         author = self._extract_tag(config['author_prop'], tags=['article'], attr='data-author')
      else:
         author = self._extract_tag(config['author_prop'])

      paragraphs = self._extract_paragraphs(config['paragraph_selector'])

      # Update the result dictionary with the extracted data
      self.result.update({
         'author' : author,
         'copyright' : copyright,
         'date_published' : date_published,
         'date_modified' : date_modified,
         'language' : language,
         'source' : source,
         'title' : title,
         'description' : description,
         'paragraphs' : paragraphs,
         'url' : url,
         'image' : image,
         'label' : '...'  # Placeholder for label
      })
      return self.result


   def run_and_write(self, urls: List[str], folder: str = "Data/") -> None:
      if urls is None or not isinstance(urls, list):
         raise ValueError("URL must be a non-empty list of strings.")
      
      if not os.path.exists(folder):
         os.mkdir(folder)

      amount = sum(1 for entry in os.scandir(folder) if entry.is_file())
      for link in urls:
         self(link) # this works because __call__ is defined
         file_name = f'{self.type}_{amount + 1}'
         self.WriteJSON(folder, file_name)
         amount += 1


   def WriteJSON(self, Path: str, file_name: str) -> None:
      if not os.path.exists(Path):
         os.makedirs(Path)
      
      file_path = os.path.join(Path, file_name + ".json")
      with open(file_path, "w") as f:
         json.dump(self.result, f, indent=4, ensure_ascii=False)
         print(f"Data written to {file_path}")

   
   @staticmethod
   def _get(url: str) -> str:
      res = Scrappers.session.get(url)
      if res.status_code == 200:
            res.encoding = 'utf-8'
            return res.text
      else:
            raise Exception(f"Failed to fetch data from {url}, status code: {res.status_code}")


   @staticmethod
   def _determine_type(url: str) -> str:
      if "vnexpress" in url:
            return "vnexpress"
      elif "soha.vn" in url:
            return "soha"
      elif "tuoitre.vn" in url:
            return "tuoitre"
      elif "thanhnien.vn" in url:
            return "thanhnien"
      elif "kenh14.vn" in url:
            return "kenh14"
      elif "gamek.vn" in url:
            return "gamek"
      elif "theanh28.vn" in url:
            return "theanh28"
      elif "chinhphu.vn" in url:
            return "chinhphu"
      elif "vietnamnet.vn" in url:
            return "vietnamnet"
      elif "nld.com.vn" in url:
            return "nld"
      elif "dantri.com.vn" in url:
            return "dantri"
      else:
            raise ValueError("Unknown type: " + url)
        

   @staticmethod
   def _clean_text(text: str) -> str:
      lines = text.splitlines()
      cleaned_lines = []
      for line in lines:
            # Remove lines starting with 'ẢNH:' or similar tags
            if re.match(r'^\s*ẢNH\s*[:\-]', line, re.IGNORECASE):
               continue
            # Add space around Vietnamese text that might be stuck together
            line = re.sub(r'([a-zA-ZÀ-ỹ])([A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ])', r'\1 \2', line)
            cleaned_lines.append(line)
      return "\n".join(cleaned_lines).strip()
   

   def _extract_json_ld_data(self, key: str, default_value: str = None) -> str:
      script_tag = self.bs.find('script', {'type': 'application/ld+json'})
      if script_tag:
         try:
               json_data = json.loads(script_tag.string.strip())
               return json_data.get(key, default_value)
         except (json.JSONDecodeError, AttributeError):
               return default_value
      return default_value
      

   def __call__(self, url: str | List[str], folder: str = None, file_name: str = None) -> Any: # type:ignore
      if not url:
            raise ValueError("URL must be a non-empty string or list of strings.")
      
      if isinstance(url, list):
         if folder is None:
               raise ValueError("Folder must be specified when passing a list of URLs.")
         return self.run_and_write(url, folder)
         
      html = self._get(url)
      self.bs = BeautifulSoup(html, 'lxml')
      self.type = self._determine_type(url)

      if self.type == 'theanh28':
         # Disable SSL verification for the session
          self.session.verify = False
      else:
          self.session.verify = True

      return self.scrape(self.type)  # Convert type to lowercase to match SITE_CONFIG keys