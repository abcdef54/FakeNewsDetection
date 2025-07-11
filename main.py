from scrapper import Scrappers

if __name__ == "__main__":
    # How to use the Scrappers class
    # Option 1: Single URL
    url = "https://vnexpress.net/duong-day-san-xuat-xe-may-dien-gia-bi-phat-hien-4913042.html"
    folder = "TestSingleURL/"
    sc = Scrappers()
    sc(url)
    sc.WriteJSON(folder, "single_url_result")  # Folder to save the JSON file, and the name of the file

    # Option 2: List of URLs
    urls = [
        # Add your URLs here
        "https://vnexpress.net/cuu-dai-su-noi-ve-cau-noi-giup-binh-thuong-hoa-quan-he-viet-my-4912958.html",
        "https://vnexpress.net/duong-day-san-xuat-xe-may-dien-gia-bi-phat-hien-4913042.html",
        "https://vnexpress.net/noi-tuyet-vong-tren-nhung-tau-hang-bi-houthi-vay-khon-o-bien-do-4912730.html"
    ]
    folder = "TestListURLs/"
    sc = Scrappers()
    sc(urls, folder) # Specify folder for list of URLs 