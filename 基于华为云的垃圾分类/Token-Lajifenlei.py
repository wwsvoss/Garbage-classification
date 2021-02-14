# -*- coding: utf-8 -*-
#!/usr/bin/python3
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import sys
import json
import requests
from cv2 import cv2

token_url = "https://iam.cn-north-4.myhuaweicloud.com/v3/auth/tokens"
ModelArts_url = 'https://ebe127c49e07432ea7b7f3a4e4fab1b2.apig.cn-north-4.huaweicloudapis.com/v1/infers/e25263b6-4532-48ab-801f-5c48d1ce4714'

def getToken():
    HWdata = {
            "auth": {
                "identity": {
                "methods": ["password"],
                "password": {
                    "user": {
                    "name": name,  
                    "password": password,
                    "domain": {"name": name}
                            }
                        }
                    },
                "scope": {"project":{"name":"cn-north-4"}}
            }
        }
    HWheaders = {"Content-Type": "application/json"}
    response = requests.post(token_url, data=json.dumps(HWdata), headers=HWheaders)
    # print(response.status_code)
    if response.status_code == 201:
        token = response.headers["X-Subject-Token"]
        return token
    else:
        return ''

def writetxt(txt):
    fw = open("C:/VisionMaster/config/token.txt",'w')
    fw.write(txt)
    fw.close()

def imgpro(image_source):
    dst = cv2.cvtColor(image_source,cv2.COLOR_RGB2GRAY)
    ret,dst_2=cv2.threshold(dst,90,230,cv2.THRESH_BINARY)
    # cv2.imshow("frame", dst_2)
    # cv2.waitKey(0)  

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(dst_2, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("frame", opening)
    # cv2.waitKey(0)  

    # 找轮廓 opencv-3.4
    # binary,contours,hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 找轮廓 opencv-4.2
    contours,binary = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("frame", opening)
    # cv2.waitKey(0)  

    src_roi = image_source
    for i in range(0,len(contours)):
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        area = cv2.contourArea(contours[i])
        # print(area)
        if area < 20000:
            # 当轮廓面积小于20000时跳过，继续寻找轮廓
            continue  
        rot_img = cv2.getRotationMatrix2D(rect[0], rect[2], 1.0)
        cv2.drawContours(image_source, [box], 0, (0, 255, 0), 1)
        img_waf = cv2.warpAffine(image_source, rot_img, (image_source.shape[0],image_source.shape[1]))
        src_roi = img_waf[int(rect[0][1])-int((rect[1][1])/2)+4:int(rect[0][1])+int((rect[1][1])/2)-4,\
                          int(rect[0][0])-int((rect[1][0])/2)+4:int(rect[0][0])+int((rect[1][0])/2)-4]    
        break
    return src_roi 

def readtxt(textPath):
    with open(textPath,"r",encoding='utf-8') as f:
        # 一次性读全部成一个字符串   
        ftext = f.read()
    return ftext

def onlinepre():
    with open('C:/VisionMaster/img.jpg', "rb") as fp:
        byteimg = fp.read()
    token = readtxt(("C:/VisionMaster/config/token.txt"))
    headers = {"X-Auth-Token": token}
    file = {"images": byteimg}
    res = requests.post(ModelArts_url, files=file, headers=headers)
    # print(res.status_code)
    resp = res.json()
    json_str = json.dumps(resp)
    data_dict = json.loads(json_str)
    print(data_dict)
    text = ''
    predictedResult = ''
    if 'predicted_label' in data_dict:
        text = data_dict["predicted_label"]  
        predictedResult = text
        print('推理结果为:',predictedResult)
    return text

def showResult(text):
    img = cv2.imread('C:/VisionMaster/image.jpg')
    newimg = cv2.resize(img,(640,480))
    fontpath = "font/simsun.ttc"
    font = ImageFont.truetype(fontpath, 48)  
    img_pil = Image.fromarray(newimg)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 100), text, font = font, fill = (0, 0, 255))
    bk_img = np.array(img_pil)
    cv2.imshow('frame', bk_img)
    key = cv2.waitKey(5000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
      while True:
        name = str(input('请输入华为云账号:'))
        password = getpass.getpass(input('请输入华为云密码:'))   
        token = getToken()
        if len(token)!=0:
            writetxt(token)
            print('已成功写入Token鉴权，有效期为24小时.\n')
            image_source = cv2.imread('C:/VisionMaster/image.jpg')
            cv2.imshow("frame", image_source)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            img = imgpro(image_source)
            cv2.imwrite('C:/VisionMaster/img.jpg',img)
            text = onlinepre()
            showResult(text)
            break
        else:
            print('用户名或密码有误，请重新输入!\n')