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


   def __init__(self) -> None:
      self.bs = None
      self.type: str = ""
      self.result: Dict[str, Any] = {}

   
   def _extract_tag(self, prop=None, name=None, itemprop=None, tag_name: str = "meta",
                     default_value=None, attribute: str = 'content') -> str | None:
      
      # selector = {'property' : prop, 'name' : name, 'itemprop' : itemprop} if prop or name or itemprop else {}
      selector = {'property': prop} if prop else {}
      if name:
         selector = {'name': name}
      if itemprop:
         selector = {'itemprop': itemprop}
      
      tag = self.bs.find(tag_name, selector) # type:ignore
      if not tag:
         return default_value
      
      return tag.get(attribute, default_value) if default_value is not None else tag.get(attribute) # type:ignore
   
   def _extract_paragraphs(self, selector: str) -> str:
      paragraph_tags = self.bs.select_one(selector) # type:ignore
      if not paragraph_tags:
         raise ValueError("No paragraph tags found in the HTML content.")
      
      p_tags = paragraph_tags.find_all('p') 
      if not p_tags:
         raise ValueError("No paragraph tags found in the specified selector.")
      
      return self._clean_text('\n'.join([p.get_text(strip=True, separator=' ') for p in p_tags])) if p_tags else 'No paragraphs found' # type:ignore


   def scrape(self, source, paragraph_selector, date_published_prop, date_modified_prop, author_prop) -> Dict[str, Any]:
      title = self._extract_tag(prop="og:title", default_value='No title found', tag_name="meta", attribute='content')
      url = self._extract_tag(prop="og:url", default_value='No URL found', tag_name="meta", attribute='content')
      image = self._extract_tag(prop="og:image", default_value='No image found', tag_name="meta", attribute='content')
      description = self._extract_tag(prop="og:description", default_value='No description found', tag_name="meta", attribute='content') 

      date_published = self._extract_tag(prop=date_published_prop, tag_name="meta", attribute='content') \
      or self._extract_tag(itemprop=date_modified_prop, default_value='No date published found', tag_name='meta', attribute='content')

      date_modified = self._extract_tag(prop=date_modified_prop, tag_name="meta", attribute='content') \
      or self._extract_tag(itemprop=date_modified_prop, tag_name='meta', attribute='content') \
      or self._extract_tag(name=date_modified_prop, default_value='No date modified found', tag_name='input', attribute='value')

      author = self._extract_tag(prop=author_prop, default_value=self.type, tag_name="meta", attribute='content')
      copyright = self._extract_tag(name='copyright', default_value='No copyright found', tag_name="meta", attribute='content')
      paragraphs = self._extract_paragraphs(paragraph_selector)

      # Try different selectors for language
      language = \
      self._extract_tag(prop='og:locale', tag_name="meta", attribute='content') \
      or self._extract_tag(name='Language', tag_name="meta", attribute='content') \
      or self._extract_tag(itemprop='inLanguage', tag_name="meta", attribute='content')

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
         'label' : '...'  # Placeholder for label, can be updated later
      })
      return self.result
        

   def VNExpress(self) -> Dict[str, Any]:
      return self.scrape(
         source='vnexpress',
         paragraph_selector='article.fck_detail',
         date_published_prop='datePublished',
         date_modified_prop='dateModified',
         author_prop='authorInfo'
      )


   def TuoiTre(self) -> Dict[str, Any]:
      return self.scrape(
         source='tuoitre',
         paragraph_selector='div[itemprop="articleBody"]',
         date_published_prop='article:published_time',
         date_modified_prop='article:modified_time',
         author_prop='dable:author'
      )


   def ThanhNien(self) -> Dict[str, Any]:
      return self.scrape(
         source='thanhnien',
         paragraph_selector='div[itemprop="articleBody"]',
         date_published_prop='datePublished',
         date_modified_prop='dateModified',
         author_prop='dable:author'
      )


   def Kenh14(self) -> Dict[str, Any]:
      return self.scrape(
         source='kenh14',
         paragraph_selector='div.detail-content.afcbc-body',
         date_published_prop='article:published_time',
         date_modified_prop='hideLastModifiedDate',
         author_prop='article:author'
      )


   def Soha(self) -> Dict[str, Any]:
      return self.scrape(
         source='soha',
         paragraph_selector='div.detail-content.afcbc-body',
         date_published_prop='article:published_time',
         date_modified_prop='hidLastModifiedDate',
         author_prop='article:author'
      )


   def GameK(self) -> Dict[str, Any]:
      pass


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
   def get(url: str) -> str:
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
            return "Soha"
      elif "tuoitre.vn" in url:
            return "TuoiTre"
      elif "thanhnien.vn" in url:
            return "ThanhNien"
      elif "kenh14.vn" in url:
            return "Kenh14"
      elif "gamek.vn" in url:
            return "GameK"
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
      

   def __call__(self, url: str | List[str], folder: str = None, file_name: str = None) -> Any: # type:ignore
      if not url:
            raise ValueError("URL must be a non-empty string or list of strings.")
      
      if isinstance(url, list):
         if folder is None:
               raise ValueError("Folder must be specified when passing a list of URLs.")
         return self.run_and_write(url, folder)
         
      html = self.get(url)
      self.bs = BeautifulSoup(html, 'lxml')
      self.type = self._determine_type(url)

      if self.type == 'vnexpress':
         return self.VNExpress()
      elif self.type == 'Soha':
         return self.Soha()
      elif self.type == 'TuoiTre':
         return self.TuoiTre()
      elif self.type == 'ThanhNien':
         return self.ThanhNien()
      elif self.type == 'Kenh14':
         return self.Kenh14()
      else:
         raise ValueError(f"Unknown type: {type}")