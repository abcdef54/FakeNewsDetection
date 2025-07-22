from scrapper import Scrappers
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if __name__ == "__main__":
    # # How to use the Scrappers class
    # url = "https://vnexpress.net/duong-day-san-xuat-xe-may-dien-gia-bi-phat-hien-4913042.html"

    # # Get to know the parameters
    # # Case 1: Word limit
    # sc = Scrappers(word_limit=100)  # Set word limit to 100
    # sc(url) 
    # sc.WriteJSON("Scrapper2/", "WordLimit_test")

    # # Case 2: Paragraphs
    # sc = Scrappers(paragraphs=3)  # Set to extract 3 paragraphs
    # sc(url)
    # sc.WriteJSON("Scrapper2/", "Paragraphs_test")

    # # Case 3: Random paragraphs
    # # Take random only works when paragraphs is specified and word_limit is not specified.
    # sc = Scrappers(paragraphs=3, take_random=True)  # Set to extract 3 random paragraphs
    # sc(url)
    # sc.WriteJSON("Scrapper2/", "RandomParagraphs_test")

    # # Case 4: Word limit and paragraphs
    # # If both word_limit and paragraphs are specified, word_limit will take precedence.
    # # Because paragraphs will be disabled so will take_random even if specified.
    # sc = Scrappers(word_limit=100, paragraphs=3, take_random = True)  # Set word limit to 100 and paragraphs to 3
    # sc(url)
    # sc.WriteJSON("Scrapper2/", "WordLimitAndParagraphs_test")

    # # Default case: Extract all paragraphs
    # sc = Scrappers()  # No word limit or paragraphs specified
    # sc(url)
    # sc.WriteJSON("Scrapper2/", "Default_test")

    # Example usage of Scrappers class with a single URL or a list of URLs
    # # Option 1: Single URL
    # folder = "TestSingleURL/"
    # sc = Scrappers()
    # sc(url)
    # sc.WriteJSON(folder, "single_url_result")  # Folder to save the JSON file, and the name of the file

    # Option 2: List of URLs
    urls = [
        # Add your URLs here

    ]
    folder = "Data/"
    sc = Scrappers(word_limit=150)
    sc(urls, folder) # Specify folder for list of URLs 
