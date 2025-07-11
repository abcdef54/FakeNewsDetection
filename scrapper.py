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
   

   def VNExpress(self) -> Dict[str, Any]:
         title_tag = self.bs.find("meta", property="og:title") # type:ignore
         if not title_tag:
            raise ValueError("Title tag not found in the HTML content.")
         
         paragraph_tags = self.bs.find_all("p", class_="Normal") # type:ignore
         if not paragraph_tags:
            raise ValueError("No paragraph tags found in the HTML content.")

         source = "vnexpress"

         url = self.bs.find("meta", property="og:url").get('content', 'No URL found')

         image_tag = self.bs.find("meta", property="og:image")
         if not image_tag:
            raise ValueError("Image tag not found in the HTML content.")
         
         description_tag = self.bs.find("meta", property="og:description")
         if not description_tag:
            raise ValueError("Description tag not found in the HTML content.")
         
         date_published = self.bs.find("meta", itemprop="datePublished")
         if not date_published:
            raise ValueError("Date published tag not found in the HTML content.")
         
         date_modified = self.bs.find("meta", itemprop="dateModified")
         if not date_modified:
            raise ValueError("Date modified tag not found in the HTML content.")
         
         date_created = self.bs.find("meta", itemprop="dateCreated")
         if not date_created:
            raise ValueError("Date created tag not found in the HTML content.")
         
         author_tag =  self.bs.find("meta", {'name' : 'authorInfo'})
         if not author_tag:
            author_tag = {'content' : 'No author found'}
         else:
            author_tag = {'content': author_tag.get('content')}
         
         copyright_tag = self.bs.find("meta", {'name' : 'copyright'})
         if not copyright_tag:
            raise ValueError("Copyright tag not found in the HTML content.")
         
         language_tag = self.bs.find("meta", itemprop='inLanguage')
         if not language_tag:
            raise ValueError("Language tag not found in the HTML content.")
         
         # Update the result dictionary with the extracted data
         self.update_result(title_tag, paragraph_tags, source, url, image_tag, description_tag,
                           date_published, date_modified, author_tag, language_tag, copyright_tag)

         return self.result
      

   def TuoiTre(self) -> Dict[str, Any]:
      title_tag = self.bs.find("meta", property="og:title") # type:ignore
      if not title_tag:
         raise ValueError("Title tag not found in the HTML content.")
      
      paragraph_tags = self.bs.find('div', itemprop='articleBody').find_all('p') # type:ignore
      if not paragraph_tags:
         raise ValueError("No paragraph tags found in the HTML content.")

      source = "tuoitre"

      url = self.bs.find("meta", property="og:url").get('content', 'No URL found')

      image_tag = self.bs.find("meta", property="og:image")
      if not image_tag:
         raise ValueError("Image tag not found in the HTML content.")
      
      description_tag = self.bs.find("meta", property="og:description")
      if not description_tag:
         raise ValueError("Description tag not found in the HTML content.")
      
      date_published = self.bs.find("meta", property="article:published_time")
      if not date_published:
         raise ValueError("Date published tag not found in the HTML content.")
      
      date_modified = self.bs.find("meta", property="article:modified_time")
      if not date_modified:
         raise ValueError("Date modified tag not found in the HTML content.")
      
      author_tag =  self.bs.find("meta", property="dable:author")
      if not author_tag:
         raise ValueError("Author tag not found in the HTML content.")
      
      copyright_tag = self.bs.find("meta", {'name' : 'copyright'})
      if not copyright_tag:
         raise ValueError("Copyright tag not found in the HTML content.")
      
      language_tag = self.bs.find("meta", property='og:locale')
      if not language_tag:
         raise ValueError("Language tag not found in the HTML content.")
      
      # Update the result dictionary with the extracted data
      self.update_result(title_tag, paragraph_tags, source, url, image_tag, description_tag,
                           date_published, date_modified, author_tag, language_tag, copyright_tag)

      return self.result
   

   def ThanhNien(self) -> None:
      title_tag = self.bs.find("meta", property="og:title") # type:ignore
      if not title_tag:
         raise ValueError("Title tag not found in the HTML content.")
      
      # <div data-check-position="body_start"></div>
      paragraph_tags = self.bs.find('div', itemprop='articleBody').find_all('p') # type:ignore
      if not paragraph_tags:
         raise ValueError("No paragraph tags found in the HTML content.")

      source = "thanhnien"

      url = self.bs.find("meta", property="og:url").get('content', 'No URL found')

      image_tag = self.bs.find("meta", property="og:image")
      if not image_tag:
         raise ValueError("Image tag not found in the HTML content.")
      
      description_tag = self.bs.find("meta", property="og:description")
      if not description_tag:
         raise ValueError("Description tag not found in the HTML content.")
      
      date_published = self.bs.find("meta", itemprop="datePublished")
      if not date_published:
         raise ValueError("Date published tag not found in the HTML content.")
      
      date_modified = self.bs.find("meta", itemprop="dateModified"   )
      if not date_modified:
         raise ValueError("Date modified tag not found in the HTML content.")
      
      author_tag =  self.bs.find("meta", property="dable:author")
      if not author_tag:
         raise ValueError("Author tag not found in the HTML content.")
      
      copyright_tag = self.bs.find("meta", {'name' : 'copyright'})
      if not copyright_tag:
         raise ValueError("Copyright tag not found in the HTML content.")
      
      language_tag = self.bs.find('html', lang=True)
      if not language_tag:
         raise ValueError("Language tag not found in the HTML content.")
      else:
         language_tag = {'content': language_tag.get('lang', 'No language found')}
      
      # Update the result dictionary with the extracted data
      self.update_result(title_tag, paragraph_tags, source, url, image_tag, description_tag,
                           date_published, date_modified, author_tag, language_tag, copyright_tag)

      return self.result
   

   def Kenh14(self) -> None:
      title_tag = self.bs.find("meta", property="og:title") # type:ignore
      if not title_tag:
         raise ValueError("Title tag not found in the HTML content.")
      
      paragraph_tags = self.bs.find('div', class_='detail-content afcbc-body').find_all('p') # type:ignore
      if not paragraph_tags:
         raise ValueError("No paragraph tags found in the HTML content.")

      source = "kenh14"

      url = self.bs.find("meta", property="og:url").get('content', 'No URL found')

      image_tag = self.bs.find("meta", property="og:image")
      if not image_tag:
         raise ValueError("Image tag not found in the HTML content.")
      
      description_tag = self.bs.find("meta", property="og:description")
      if not description_tag:
         raise ValueError("Description tag not found in the HTML content.")
      
      date_published = self.bs.find("meta", property="article:published_time")
      if not date_published:
         raise ValueError("Date published tag not found in the HTML content.")
      
      date_modified = self.bs.find("input", {'name' : 'hideLastModifiedDate'})
      if not date_modified:
         raise ValueError("Date modified tag not found in the HTML content.")
      else:
         date_modified = {'content': date_modified.get('value', 'No date modified found')}
      
      author_tag =  self.bs.find("meta", property="article:author")
      if not author_tag:
         raise ValueError("Author tag not found in the HTML content.")
      
      copyright_tag = self.bs.find("meta", {'name' : 'copyright'})
      if not copyright_tag:
         raise ValueError("Copyright tag not found in the HTML content.")
      
      language_tag = self.bs.find("meta", {'name' : 'Language'})
      if not language_tag:
         raise ValueError("Language tag not found in the HTML content.")
      
      # Update the result dictionary with the extracted data
      self.update_result(title_tag, paragraph_tags, source, url, image_tag, description_tag,
                           date_published, date_modified, author_tag, language_tag, copyright_tag)

      return self.result


   def FaceBook(self) -> ...:
        pass
      
   

   def update_result(self, title_tag, paragraph_tags, source, url, image_tag, description_tag,
                     date_published, date_modified, author_tag, language_tag, copyright_tag) -> None:
      self.result['author'] = author_tag.get('content', 'No author found')
      self.result['copyright'] = copyright_tag.get('content', 'No copyright found')
      self.result['date_published'] = date_published.get('content', 'No date published found')
      self.result['date_modified'] = date_modified.get('content', 'No date modified found')
      self.result['language'] = language_tag.get('content', 'No language found')
      self.result['title'] = title_tag.get('content', 'No title found')
      self.result['paragraphs'] = self._clean_text('\n'.join([p.get_text(strip=True) for p in paragraph_tags])) if paragraph_tags else 'No paragraphs found'
      self.result['source'] = source
      self.result['url'] = url
      self.result['image'] = image_tag.get('content', 'No image found')
      self.result['description'] = description_tag.get('content', 'No description found')

   
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
      elif "facebook.com" in url:
         return "FaceBook"
      elif "tuoitre.vn" in url:
         return "TuoiTre"
      elif "thanhnien.vn" in url:
         return "ThanhNien"
      elif "kenh14.vn" in url:
         return "Kenh14"
      else:
         return "Custom_Data"
   

   @staticmethod
   def _clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Remove lines starting with 'ẢNH:' or similar tags
        if re.match(r'^\s*ẢNH\s*[:\-]', line, re.IGNORECASE):
            continue
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
     elif self.type == 'FaceBook':
        return self.FaceBook()
     elif self.type == 'TuoiTre':
        return self.TuoiTre()
     elif self.type == 'ThanhNien':
        return self.ThanhNien()
     elif self.type == 'Kenh14':
        return self.Kenh14()
     else:
        raise ValueError(f"Unknown type: {type}")