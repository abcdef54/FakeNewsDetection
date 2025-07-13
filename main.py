from scrapper import Scrappers
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if __name__ == "__main__":
    # # How to use the Scrappers class
    # # Option 1: Single URL
    # url = "https://vnexpress.net/duong-day-san-xuat-xe-may-dien-gia-bi-phat-hien-4913042.html"
    # folder = "TestSingleURL/"
    # sc = Scrappers()
    # sc(url)
    # sc.WriteJSON(folder, "single_url_result")  # Folder to save the JSON file, and the name of the file

    # Option 2: List of URLs
    urls = [
        # Add your URLs here
        "https://vnexpress.net/cuu-dai-su-noi-ve-cau-noi-giup-binh-thuong-hoa-quan-he-viet-my-4912958.html",
        "https://tuoitre.vn/thu-tuong-muon-can-tho-tien-phong-ve-khoa-hoc-cong-nghe-doi-moi-sang-tao-va-chuyen-doi-so-2025071318184971.htm",
        "https://vietnamnet.vn/tong-thong-iran-bi-thuong-trong-cuoc-khong-kich-cua-israel-2421204.html",
        "https://thanhnien.vn/bo-y-te-de-xuat-hon-151-ti-ho-tro-cac-gia-dinh-chi-sinh-2-con-gai-185250713181742206.htm",
        "https://kenh14.vn/khach-viet-thang-thot-khi-goi-xien-nuong-ven-duong-o-trung-quoc-toi-khong-biet-nuot-kieu-gi-luon-215250713195229459.chn",
        "https://soha.vn/nu-nghe-si-dinh-dam-phai-roi-showbiz-vi-clip-chan-dong-u50-le-bong-khong-con-cai-o-xu-nguoi-198250713165450469.htm",
        "https://theanh28.vn/threads/tran-nhat-tuan-giam-doc-chien-luoc-theanh28-entertainment-va-chuyen-khoi-nghiep.3633/",
        "https://gamek.vn/tua-game-battle-royale-lay-cam-hung-tu-gta-doi-ten-lan-thu-hai-tiep-tuc-mien-phi-tren-steam-178250710101437849.chn",
        "https://baochinhphu.vn/no-luc-cao-nhat-phan-dau-hoan-thanh-thang-loi-cac-chi-tieu-phat-trien-kt-xh-nam-2025-102250712113541294.htm",
        "https://laodong.vn/van-hoa-giai-tri/unesco-cong-nhan-di-san-the-gioi-lien-bien-gioi-dau-tien-giua-viet-nam-va-lao-1539552.ldo",
        "https://dantri.com.vn/xa-hoi/don-doc-tphcm-trinh-phuong-an-han-che-xe-phat-thai-cao-20250713180551692.htm",
    ]
    folder = "TestListURLs/"
    sc = Scrappers()
    sc(urls, folder) # Specify folder for list of URLs 