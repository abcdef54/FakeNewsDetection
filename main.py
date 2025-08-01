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
    # # Take random works for both word_limit and paragraphs
    # sc = Scrappers(paragraphs=3, take_random=True)  # Set to extract 3 random paragraphs that are next to each other
    # sc = Scrappers(word_limit=100, take_random=True)  # Set to extract 100 random words that are next to each other
    # sc(url)
    # sc.WriteJSON("Scrapper2/", "RandomParagraphs_test")


    # # Case 4: Word limit and paragraphs
    # # If both word_limit and paragraphs are specified, word_limit will take precedence.
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
        # "https://vnexpress.net/cuu-dai-su-noi-ve-cau-noi-giup-binh-thuong-hoa-quan-he-viet-my-4912958.html",
        # "https://tuoitre.vn/thu-tuong-muon-can-tho-tien-phong-ve-khoa-hoc-cong-nghe-doi-moi-sang-tao-va-chuyen-doi-so-2025071318184971.htm",
        # "https://vietnamnet.vn/tong-thong-iran-bi-thuong-trong-cuoc-khong-kich-cua-israel-2421204.html",
        # "https://thanhnien.vn/bo-y-te-de-xuat-hon-151-ti-ho-tro-cac-gia-dinh-chi-sinh-2-con-gai-185250713181742206.htm",
        # "https://kenh14.vn/khach-viet-thang-thot-khi-goi-xien-nuong-ven-duong-o-trung-quoc-toi-khong-biet-nuot-kieu-gi-luon-215250713195229459.chn",
        # "https://soha.vn/nu-nghe-si-dinh-dam-phai-roi-showbiz-vi-clip-chan-dong-u50-le-bong-khong-con-cai-o-xu-nguoi-198250713165450469.htm",
        # "https://theanh28.vn/threads/tran-nhat-tuan-giam-doc-chien-luoc-theanh28-entertainment-va-chuyen-khoi-nghiep.3633/",
        # "https://gamek.vn/tua-game-battle-royale-lay-cam-hung-tu-gta-doi-ten-lan-thu-hai-tiep-tuc-mien-phi-tren-steam-178250710101437849.chn",
        # "https://baochinhphu.vn/no-luc-cao-nhat-phan-dau-hoan-thanh-thang-loi-cac-chi-tieu-phat-trien-kt-xh-nam-2025-102250712113541294.htm",
        # "https://laodong.vn/van-hoa-giai-tri/unesco-cong-nhan-di-san-the-gioi-lien-bien-gioi-dau-tien-giua-viet-nam-va-lao-1539552.ldo",
        # "https://dantri.com.vn/xa-hoi/don-doc-tphcm-trinh-phuong-an-han-che-xe-phat-thai-cao-20250713180551692.htm",
        # 'https://tingia.gov.vn/bat-coc-hue.html',
        # "https://vietgiaitri.com/bi-an-hop-dong-hon-nhan-cua-ronaldo-va-ban-gai-sexy-nang-wag-duoc-chu-cap-tien-ty-mot-thang-neu-chia-tay-20250728i7497452/",
        # "https://www.webtretho.com/f/chuan-bi-mang-thai/chuan-bi-mang-thai-tu-nhien-khong-don-gian-la-ngung-tranh-thai",
        # "https://tiin.vn/chuyen-muc/song/pham-thoai-xuat-hien-tieu-tuy-sau-on-ao-sao-ke.html",
        # "https://www.24h.com.vn/tin-tuc-quoc-te/thai-lan-len-tieng-thong-tin-danh-trung-toa-nha-casino-o-bien-gioi-campuchia-c415a1684664.html",
        "https://baoangiang.com.vn/sieu-lua-an-do-dung-dai-su-quan-gia-de-chiem-doat-tien-ty-a425143.html",
    ]
    folder = "Data/"
    sc = Scrappers(word_limit=150, take_random=False)
    sc(urls, folder) # Specify folder for list of URLs 
    '''
    New updates:

    1 - Create new empty json file for social news
    Scrappers.empty_social_json(destination='Data/', source='facebook', amount=5)
    this will create 5 new empty JSON file with the correct structure you only have to put in your own text and info

    2 - Now support "take_random" parameter for both word_limit and paragraphs.
    sc = Scrappers(word_limit=100, take_random=True)  # Set to extract 100 random words that are next to each other
    sc = Scrappers(paragraphs=3, take_random=True)  # Set to extract 3 random paragraphs that are next to each other

    3 - Support 6 new news websites:
    - tingia.gov.vn
    - baoangiang.com.vn
    - vietgiaitri.com
    - webtretho.com
    - tiin.vn
    - 24h.com.vn

    All supported websites:
    [vnexpress.net, tuoitre.vn, vietnamnet.vn, thanhnien.vn, kenh14.vn, soha.vn, 
    theanh28.vn, gamek.vn, chinhphu.vn, laodong.vn, dantri.com.vn, tingia.gov.vn,
    vietgiaitri.com, webtretho.com, tiin.vn, 24h.com.vn, baoangiang.com.vn]
    '''
