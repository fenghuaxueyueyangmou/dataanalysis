import requests
import lxml
import parsel
from bs4 import BeautifulSoup
import re
import pandas
import numpy
import csv
import time

url_list = []
url_list_last = []

for i in range(100):

    url = f'https://www.che168.com/guizhou/a0_0msdgscncgpi1ltocsp{i}exx0/?pvareaid=102179#currengpostion'
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
               'Cookie':'fvlid=1709791029637qc2njzVfZcVl; sessionid=ec3deac0-bdbb-4a86-ac77-4dedb6531d67; che_sessionid=3CFC5E38-63C1-4EA7-B7B5-D2107DC3DA37%7C%7C2024-03-07+13%3A57%3A08.930%7C%7Cwww.autohome.com.cn; smidV2=20240307135720dc0ac0a2a21b9b36f5ea16c3ca516bfd00e36ae1cf53ccde0; pcpopclub=3ce1973fc8134c29820067a0d562b00a107a7fa5; clubUserShow=276463525|0|500100|%e5%8d%97%e9%98%b3%e8%bd%a6%e5%8f%8b4148222|0|0|0|/g30/M06/0C/42/120X120_0_q87_autohomecar__CjIFU2T-zw2AQTDjAAB0HWW9gzw677.png|2024-03-07 13:57:47|0; memberPhoneInfo=%7C15324766692%7C1; area=441999; UsedCarBrowseHistory=0%3A50360781%2C0%3A50375193%2C0%3A50337830%2C0%3A50326450%2C0%3A49922441%2C0%3A50375446%2C0%3A50375386%2C0%3A50266566%2C0%3A50186544%2C0%3A50220270; usedcaruid=oKRjYlTB6azjNtEOoM9VmQ==; href=https%3A%2F%2Fwww.che168.com%2Fchongqing%2F%23pvareaid%3D100943; accessId=7a783820-ec84-11ec-b95f-79694d4df285; pageViewNum=1; Hm_lvt_d381ec2f88158113b9b76f14c497ed48=1710233756,1710246648,1711521260,1711535192; sessionvisit=14585580-a51b-4403-9103-1a2d07a29165; sessionvisitInfo=ec3deac0-bdbb-4a86-ac77-4dedb6531d67|www.che168.com|0; che_sessionvid=3816E764-C65D-410A-8CA2-F5683BF95805; userarea=510100; listuserarea=510100; ahpvno=11; ahuuid=77E68501-9D78-4523-BB09-BCC1940E4578; Hm_lpvt_d381ec2f88158113b9b76f14c497ed48=1711546544; showNum=123; sessionip=183.2.201.57; v_no=124; visit_info_ad=3CFC5E38-63C1-4EA7-B7B5-D2107DC3DA37||3816E764-C65D-410A-8CA2-F5683BF95805||-1||-1||124; che_ref=www.autohome.com.cn%7C0%7C110965%7C0%7C2024-03-27+21%3A35%3A45.450%7C2024-03-07+13%3A57%3A08.930; sessionuid=ec3deac0-bdbb-4a86-ac77-4dedb6531d67'}
    response = requests.get(url,headers=headers).text
    selector = parsel.Selector(response)
    detail_url_list = selector.xpath('/html/body/div[12]/div[1]/ul/li/a/@href').getall()

    print(detail_url_list)
    url_list = detail_url_list + url_list
    time.sleep(1)
print(len(url_list))

with open('car.csv','a',encoding='utf-8-sig',newline='') as f:


    row_name = ['车名','价格','里程','发布时间','转手次数','挡位','排量','驱动类型','环保标准','发动机','车型','燃油类型','颜色']
    csv_witer = csv.writer(f)
    csv_witer.writerow(row_name)
    for detail_url in url_list:
        detail_url = 'https://www.che168.com/' + detail_url
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                   'Cookie':'fvlid=1709791029637qc2njzVfZcVl; sessionid=ec3deac0-bdbb-4a86-ac77-4dedb6531d67; che_sessionid=3CFC5E38-63C1-4EA7-B7B5-D2107DC3DA37%7C%7C2024-03-07+13%3A57%3A08.930%7C%7Cwww.autohome.com.cn; smidV2=20240307135720dc0ac0a2a21b9b36f5ea16c3ca516bfd00e36ae1cf53ccde0; pcpopclub=3ce1973fc8134c29820067a0d562b00a107a7fa5; clubUserShow=276463525|0|500100|%e5%8d%97%e9%98%b3%e8%bd%a6%e5%8f%8b4148222|0|0|0|/g30/M06/0C/42/120X120_0_q87_autohomecar__CjIFU2T-zw2AQTDjAAB0HWW9gzw677.png|2024-03-07 13:57:47|0; memberPhoneInfo=%7C15324766692%7C1; UsedCarBrowseHistory=0%3A50360781%2C0%3A50375193%2C0%3A50337830%2C0%3A50326450%2C0%3A49922441%2C0%3A50375446%2C0%3A50375386%2C0%3A50266566%2C0%3A50186544%2C0%3A50220270; area=441999; Hm_lvt_d381ec2f88158113b9b76f14c497ed48=1711521260,1711535192,1711558726,1711602194; userarea=520000; listuserarea=520000; sessionvisit=26c0a5f6-258f-47f7-9d18-bfcddfab03ca; sessionvisitInfo=ec3deac0-bdbb-4a86-ac77-4dedb6531d67|www.che168.com|100943; che_sessionvid=7FDF132C-24E4-4B34-AFC5-5B9D4919A146; sessionip=183.2.201.71; ahpvno=37; Hm_lpvt_d381ec2f88158113b9b76f14c497ed48=1711631133; ahuuid=F51E9036-BD3E-45D1-B740-3F974255FEA5; showNum=149; v_no=150; visit_info_ad=3CFC5E38-63C1-4EA7-B7B5-D2107DC3DA37||7FDF132C-24E4-4B34-AFC5-5B9D4919A146||-1||-1||150; che_ref=www.autohome.com.cn%7C0%7C110965%7C0%7C2024-03-28+21%3A05%3A34.244%7C2024-03-07+13%3A57%3A08.930; sessionuid=ec3deac0-bdbb-4a86-ac77-4dedb6531d67'}
        response = requests.get(detail_url,headers=headers).text
        selector = parsel.Selector(response)
        name = selector.xpath('/html/body/div[5]/div[2]/h3/text()').getall()
        place = selector.xpath('/html/body/div[5]/div[2]/div[2]/span/text()').getall()
        mileage = selector.xpath('/html/body/div[5]/div[2]/ul/li[1]/h4/text()').getall()
        Registration_time = selector.xpath('/html/body/div[5]/div[2]/ul/li[2]/h4/text()').getall()
        Number_transfers = selector.xpath('/html/body/div[8]/div[1]/ul[2]/li[5]/text()').getall()
        Gear_position = selector.xpath('/html/body/div[8]/div[1]/ul[1]/li[3]/text()').getall()
        displacement = selector.xpath('/html/body/div[8]/div[1]/ul[1]/li[5]/text()').getall()
        Driving_method = selector.xpath('/html/body/div[8]/div[1]/ul[3]/li[5]/text()').getall()
        emission = selector.xpath('/html/body/div[8]/div[1]/ul[1]/li[4]/text()').getall()
        engine = selector.xpath('/html/body/div[8]/div[1]/ul[3]/li[1]/text()').getall()
        car_type = selector.xpath('/html/body/div[8]/div[1]/ul[3]/li[2]/text()').getall()
        energy_type = selector.xpath('/html/body/div[8]/div[1]/ul[3]/li[4]/text()').getall()
        color = selector.xpath('/html/body/div[8]/div[1]/ul[3]/li[3]/text()').getall()
        time.sleep(1)
        name = [s.strip() for s in name]
        name = str(name).replace('[','').replace(']','').replace('\'','')
        place = [s.strip() for s in place]
        place = str(place).replace('[','').replace(']','').replace('\'','').replace('¥','')
        mileage = [s.strip() for s in mileage]
        mileage = str(mileage).replace('[', '').replace(']', '').replace('\'', '').replace('万公里','')
        Registration_time = [s.strip() for s in Registration_time]
        Registration_time = str(Registration_time).replace('[', '').replace(']', '').replace('\'', '')
        Number_transfers = str(Number_transfers).replace('（以车辆登记证为准）','').replace('[', '').replace(']', '').replace('\'', '').replace('次','')
        Gear_position = [s.strip() for s in Gear_position]
        Gear_position = str(Gear_position).replace('[', '').replace(']', '').replace('\'', '')
        displacement = [s.strip() for s in displacement]
        displacement = str(displacement).replace('[', '').replace(']', '').replace('\'', '').replace('L','')
        Driving_method = [s.strip() for s in Driving_method]
        Driving_method = str(Driving_method).replace('[', '').replace(']', '').replace('\'', '').replace('前置','')
        emission = [s.strip() for s in emission]
        emission = str(emission).replace('[', '').replace(']', '').replace('\'', '')
        engine = [s.strip() for s in engine]
        engine = str(engine).replace('[', '').replace(']', '').replace('\'', '')
        engine = engine[:3]
        car_type = [s.strip() for s in car_type]
        car_type = str(car_type).replace('[', '').replace(']', '').replace('\'', '')
        energy_type = [s.strip() for s in energy_type]
        energy_type = str(energy_type).replace('[', '').replace(']', '').replace('\'', '').replace('号','')
        color = [s.strip() for s in color]
        color = str(color).replace('[', '').replace(']', '').replace('\'', '')



        data = [name,place,mileage,Registration_time,Number_transfers,Gear_position,displacement,Driving_method,emission,engine,car_type,energy_type,color]
        csv_witer.writerows([data])
        # print(emission)



f.close()