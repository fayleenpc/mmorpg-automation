import cv2
import numpy as np
import requests
import pydirectinput
import time
import pyperclip
import pytesseract
import base64
import json
from PIL import Image
import asyncio
import threading

import asyncio
from typing import Callable

class Regulator:
    def __init__(self, interval: float, f: Callable, *args, **kwargs):
        """
        Do not call the function f more than one per t seconds.
        
        interval is a time interval in seconds.
        f is a function
        *args and **kwargs are the usual suspects
        """
        self.interval = interval
        self.f = f
        self.args = args
        self.kwargs = kwargs
        self.busy = False
        
    async def __call__(self):
        if not self.busy:
            self.busy = True
            asyncio.get_event_loop().call_later(self.interval, self.done)
            self.f(*self.args, **self.kwargs)
            
    def done(self):
        self.busy = False
        
            



def _workaround_write(text):
    """
    This is a work-around for the bug in pyautogui.write() with non-QWERTY keyboards
    It copies the text to clipboard and pastes it, instead of typing it.
    """
    pyperclip.copy(text)
    pydirectinput.keyDown("ctrl")
    pydirectinput.press("v")
    pydirectinput.keyUp("ctrl")
    pyperclip.copy('')

    # Test
    # text_with_special_chars = '@/:;\\.ABCabc?!~^[]{}()'
    # pydirectinput.write(text_with_special_chars) 
    # >>> //;\.ABCabc?§¨[]5°      -> NOK
    
async def if_captcha():
    image = cv2.imread('captcha.jpg')
    image_captchaornot_prepare = cv2.imread('captcha.jpg')

    # Define the coordinates of the crop rectangle (x1, y1) is the top-left and (x2, y2) is the bottom-right
    x1, y1, x2, y2 = 611, 428, 982, 492
    x11, y11, x22, y22 = 614, 359, 985, 413
    # Crop the image using array slicing
    cropped_image = image[y1:y2, x1:x2]
    cropped_captchaornot_prepare = image_captchaornot_prepare[y11:y22, x11:x22]
    # Save or display the cropped image
    img1 = cropped_captchaornot_prepare
    img2 = cv2.imread('captchaornot_solve.jpg')
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)

    # similiar = img1.tobytes() == img2.tobytes()
    # if similiar:
    #     print('captcha must be solved')
    # else:
    #     print('no problem solving')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    error, diff = mse(img1, img2)
    print("Image matching Error between the two images:",error <= 1)
    cv2.imwrite('code_captcha.jpg', cropped_image)
    cv2.imwrite('captchaornot_prepare.jpg', cropped_captchaornot_prepare)
    # cv2.imshow('Cropped Image', cropped_image)
    # cv2.imshow('Cropped Image', cropped_captchaornot_prepare)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return error <= 1
    # return 1 <= error 
    # cv2.imshow("difference", diff)

# ocr api key e6e44f474388957
async def solve_captcha():
    # api_url = 'https://api.api-ninjas.com/v1/imagetotext'
    # image_file_descriptor = open('code_captcha.jpg', 'rb')
    # files = {'image': image_file_descriptor}
    # r = requests.post(api_url, files=files)

    

    #using apininja
    # api_url = 'https://api.api-ninjas.com/v1/imagetotext'
    # image_file_descriptor = open('code_captcha.jpg', 'rb')
    # files = {'image': image_file_descriptor}
    # r = requests.post(api_url, files=files, headers={'X-Api-Key': 'z7SfhHPfH+OA6ifNRV3DQg==Aa8Hez6Oqqpk6QSl'})

    # data = r.json()[0]['text']

    #using imagetextinfo
    # api_url = 'https://www.imagetotext.info/api/imageToText'
    # image_file_descriptor = open('code_captcha.jpg', 'rb')
    # files = {'image': image_file_descriptor}
    # r = requests.post(api_url, files=files, headers={'X-Api-Key': 'z7SfhHPfH+OA6ifNRV3DQg==Aa8Hez6Oqqpk6QSl'})

    # data = r.json()[0]['text']

    #using ocr
    
    # parsed_data = ''
    # for i in range(6):
    #     api_url = 'https://api.ocr.space/parse/image'
    #     each_name = 'code-captcha-split-'+str(i+1)+'.jpg'
    #     image_file_descriptor = open(each_name, 'rb')
    #     files = {each_name: image_file_descriptor}
    #     payload = {'apikey': 'e6e44f474388957', 'scale': True, 'isOverlayRequired': False, 'OCREngine': 2, 'isTable': False}
    #     r =  requests.post(api_url, files=files, data=payload)
    #     data = r.json()
    #     data = data['ParsedResults'][0]['ParsedText']
    #     if data == "":
    #         print("error : ", i+1)
    #     parsed_data += data
    #     print(data)
    # print(parsed_data)

    api_url = 'https://api.ocr.space/parse/image'
    image_file_descriptor = open('code_captcha.jpg', 'rb')
    files = {'code_captcha.jpg': image_file_descriptor}
    payload = {'apikey': 'e6e44f474388957', 'scale': True, 'isOverlayRequired': False, 'OCREngine': 2, 'isTable': False}
    r =  requests.post(api_url, files=files, data=payload)
    data = r.json()
    data = data['ParsedResults'][0]['ParsedText']
    # parsed_data += datajyp
    print(data)


    if data != "":
        time.sleep(0.2)
        pydirectinput.move(950, 630)
        time.sleep(0.2)
        pydirectinput.leftClick(950, 630)
        time.sleep(0.2)
        parsed_data = ''
        if len(data) == 6:
           parsed_data = data
           print(data)
        if len(data) >6:
            parsed_data = modify(data.replace(' ', '').replace('\r', '').replace('\n', ''))
            print(parsed_data)
        pydirectinput.PAUSE = 0.3
        for char in parsed_data:
            time.sleep(0.3)
            if char.isalpha():
               pydirectinput.press(char.lower())
            if char.isdigit():
               pydirectinput.press(char)
        
        pydirectinput.move(950, 665)
        time.sleep(0.2)
        pydirectinput.leftClick(950, 665)
        time.sleep(0.2)
    else:
       print("true but no solve")

                


    

def modify(string):  
  final = ''
  n_max = 6
  error = len(string)-n_max
  for i in range(len(string)):  
    
    if i % 2 == 1 and string[i] == '1' and error == 0:
       final += string[i]
    elif i % 2 == 1 and string[i] == '1':
      error -=1
      continue
    else : 
        final += string[i]
        # print(error)
  return final

# define the function to compute MSE between two images
def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff

foo = Regulator(0.5, solve_captcha)
foobar = Regulator(0.5, if_captcha)

async def scanning_captcha():
    await asyncio.gather(foobar(), foo())
    captcha = await if_captcha()
    await asyncio.sleep(1.0)
    if True:
        await asyncio.sleep(1.0)  
        await solve_captcha()

if __name__ == "__main__":
    asyncio.run(scanning_captcha())
    
        

    






