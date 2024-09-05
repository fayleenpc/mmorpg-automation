import numpy as np
import win32gui, win32ui, win32con, pyautogui, pydirectinput
from PIL import Image
from time import sleep
import cv2 as cv
import os

import math

import sys
import win32process as wproc
import win32api as wapi

import threading
import time
import requests
import pyperclip

import asyncio
from typing import Callable
# Function to simulate a time-consuming task

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


class WindowCapture:
    w = 0
    h = 0
    hwnd = None

    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

    def get_screenshot(self):
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[...,:3]
        img = np.ascontiguousarray(img) 
            
        return img

    def generate_image_dataset(self):
        if not os.path.exists("images"):
            os.mkdir("images")
        while(True):
            img = self.get_screenshot()
            im = Image.fromarray(img[..., [2, 1, 0]])
            im.save(f"./images/img_{len(os.listdir('images'))}.jpeg")
            sleep(1)
    
    def get_window_size(self):
        return (self.w, self.h)
class ImageProcessor:
    W = 0
    H = 0
    net = None
    ln = None
    classes = {}
    colors = []

    def __init__(self, img_size, cfg_file, weights_file):
        np.random.seed(42)
        self.net = cv.dnn.readNetFromDarknet(cfg_file, weights_file)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.W = img_size[0]
        self.H = img_size[1]
        
        with open('yolov4-tiny/obj.names', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            self.classes[i] = line.strip()
        
        # If you plan to utilize more than six classes, please include additional colors in this list.
        self.colors = [
            (0, 0, 255), 
            (0, 255, 0), 
            (255, 0, 0), 
            (255, 255, 0), 
            (255, 0, 255), 
            (0, 255, 255)
        ]
        

    def proccess_image(self, img):

        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)
        outputs = np.vstack(outputs)
        
        coordinates = self.get_coordinates(outputs, 0.5)

        self.draw_identified_objects(img, coordinates)

        return coordinates

    def get_coordinates(self, outputs, conf):

        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]
            
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([self.W, self.H, self.W, self.H])
                p0 = int(x - w//2), int(y - h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)

        if len(indices) == 0:
            return []

        coordinates = []
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            coordinates.append({'x': x, 'y': y, 'w': w, 'h': h, 'class': classIDs[i], 'class_name': self.classes[classIDs[i]]})
        return coordinates

    def draw_identified_objects(self, img, coordinates):
        for coordinate in coordinates:
            x = coordinate['x']
            y = coordinate['y']
            w = coordinate['w']
            h = coordinate['h']
            classID = coordinate['class']
            
            color = self.colors[classID]
            
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, self.classes[classID], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv.imshow('window',  img)
        

# Run this cell to initiate detections using the trained model.

window_name = "SoulSaverOnline"
cfg_file_name = "./yolov4-tiny/yolov4-tiny-custom.cfg"
weights_file_name = "yolov4-tiny-custom_last.weights"

wincap = WindowCapture(window_name)
improc = ImageProcessor(wincap.get_window_size(), cfg_file_name, weights_file_name)

    
# async def send_coordinate(coordinate):
#     async with websockets.connect('ws://localhost:3000/ws') as websocket:
#         await websocket.send(""+coordinate+"")
#         response = await websocket.recv()
#         print(response)

def focusing_on():
    window_handle = pyautogui.getWindowsWithTitle(window_name)[0]
    # pydirectinput.getWindowsWithTitle(window_name)[0]
    rect = window_handle._rect
    
    win32gui.SetForegroundWindow(window_handle._hWnd)
    win32gui.SetFocus(window_handle._hWnd)

def skills():

    auto_mp()
    pydirectinput.press('z')
    auto_mp()
    pydirectinput.press('x')
    auto_mp()
    pydirectinput.press('c')
    pydirectinput.press('v')
    auto_mp()
    pydirectinput.press('b')
    auto_mp()
    pydirectinput.press('n')
    auto_mp()
    
    
def auto_mp():
    pydirectinput.press('2')

def right():
    
    pydirectinput.keyDown('right')
    pydirectinput.keyDown('right')
    pydirectinput.keyDown('right')
    pydirectinput.keyUp('right')
    # pydirectinput.keyDown('right')
    # pydirectinput.keyDown('right')
    pydirectinput.press('2')
    
def left():
    pydirectinput.keyDown('left')
    pydirectinput.keyUp('left')
    pydirectinput.press('2')
    
    pydirectinput.press("pagedown")
    pydirectinput.press("`")

def jump():
    pydirectinput.keyDown('ctrl')
    pydirectinput.press('up')
    pydirectinput.keyUp('ctrl')
    
def unjump():
    pydirectinput.keyDown('ctrl')
    pydirectinput.press('down')
    pydirectinput.keyUp('ctrl')
def buff():
    pydirectinput.PAUSE = 0.3
    pydirectinput.press('4')
    pydirectinput.press('5')
    pydirectinput.press('6')
    
def awaken():
    pydirectinput.press('3')
    pydirectinput.press('delete')
    

def plaque():
    pydirectinput.press('J')
    pydirectinput.press('L')
    pydirectinput.press('end')   
    

    
async def if_captcha():
    # Load an image
    ss_captcha = wincap.get_screenshot()
    cv.imwrite('captcha.jpg', ss_captcha)
    image = cv.imread('captcha.jpg')
    image_captchaornot_prepare = cv.imread('captcha.jpg')

    # Define the coordinates of the crop rectangle (x1, y1) is the top-left and (x2, y2) is the bottom-right
    x1, y1, x2, y2 = 611, 428, 982, 492
    x11, y11, x22, y22 = 614, 359, 985, 413
    # Crop the image using array slicing
    cropped_image = image[y1:y2, x1:x2]
    cropped_captchaornot_prepare = image_captchaornot_prepare[y11:y22, x11:x22]
    # Save or display the cropped image
    img1 = cropped_captchaornot_prepare
    img2 = cv.imread('captchaornot_solve.jpg')
    # cv.imshow('img1', img1)
    # cv.imshow('img2', img2)

    # similiar = img1.tobytes() == img2.tobytes()
    # if similiar:
    #     print('captcha must be solved')
    # else:
    #     print('no problem solving')
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    error, diff = mse(img1, img2)
    print("Image matching Error between the two images:",error <= 1)
    cv.imwrite('code_captcha.jpg', cropped_image)
    cv.imwrite('captchaornot_prepare.jpg', cropped_captchaornot_prepare)
    # cv.imshow('Cropped Image', cropped_image)
    # cv.imshow('Cropped Image', cropped_captchaornot_prepare)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return error <= 1
    # return 1 <= error 
    # cv.imshow("difference", diff)

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

# ocr api key e6e44f474388957
async def solve_captcha():
    # api_url = 'https://api.api-ninjas.com/v1/imagetotext'
    # image_file_descriptor = open('code_captcha.jpg', 'rb')
    # files = {'image': image_file_descriptor}
    # r = requests.post(api_url, files=files)

    #using ocr
    api_url = 'https://api.ocr.space/parse/image'
    image_file_descriptor = open('code_captcha.jpg', 'rb')
    files = {'code_captcha.jpg': image_file_descriptor}
    payload = {'apikey': 'e6e44f474388957', 'scale': True, 'isOverlayRequired': False, 'OCREngine': 2, 'isTable': False}
    r =  requests.post(api_url, files=files, data=payload)
    data = r.json()
    data = data['ParsedResults'][0]['ParsedText']
    print(data)

    #using apininja
    # api_url = 'https://api.api-ninjas.com/v1/imagetotext'
    # image_file_descriptor = open('code_captcha.jpg', 'rb')
    # files = {'image': image_file_descriptor}
    # r = requests.post(api_url, files=files, data={'X-Api-Key': 'z7SfhHPfH+OA6ifNRV3DQg==Aa8Hez6Oqqpk6QSl'})

    # data = ''

    # if len(r.json()) == 2:
    #     data += r.json()[0]['text']
    #     data += r.json()[1]['text']
    # elif (len(r.json()) == 1):
    #     data = r.json()[0]['text']
    # data = r.json()[0]['text']
    if data != "":
        time.sleep(0.2)
        pydirectinput.move(950, 630)
        time.sleep(0.2)
        pydirectinput.leftClick(950, 630)

        parsed_data = ''
        if len(data) == 6:
           parsed_data = data
           print(data)
        if len(data) >6:
            parsed_data = modify(data.replace(' ', '').replace('\r', '').replace('\n', ''))
            print(parsed_data)
        # pydirectinput.write(parsed_data)
        pydirectinput.PAUSE = 0.3
        for char in parsed_data:
            time.sleep(0.2)
            if char.isalpha():
               pydirectinput.press(char.lower())
            if char.isdigit():
               pydirectinput.press(char)
        
        pydirectinput.move(950, 665)
        time.sleep(0.2)
        pydirectinput.leftClick(950, 665)
        time.sleep(0.2)
        buff()
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
   diff = cv.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff

def skills_hdmv3(coordinate, x, y):
    while True:
        print(coordinate)
        
        pydirectinput.press("pagedown")
        pydirectinput.press("`")
        pydirectinput.press("up")

        skills()
        
        if coordinate['x'] < 300:
            left()
            
        elif coordinate['x'] > 1000:
                 
            right()
        if coordinate['y'] > 300:
            unjump()
            skills()
        if coordinate['y'] < 300:
            jump()
            skills()
        if coordinate['y'] > 200 and coordinate['y'] < 300:
            pydirectinput.press("up")
            pydirectinput.press("up") 
            
            skills()
        elif coordinate['y'] < 200 and coordinate['y'] > 100:
            pydirectinput.press("up")
            pydirectinput.press("up") 
            skills()
        elif coordinate['y'] < 100 and coordinate['y'] > 0:
            pydirectinput.press("up")
            pydirectinput.press("up") 
            skills()

        

def check_captcha():

    
    ss_captcha = wincap.get_screenshot()
        
    coordinates = improc.proccess_image(ss_captcha)

    for coordinate in coordinates:
        # focusing_on()
        # focusing_on()
        if if_captcha(ss_captcha) == True:
            solve_captcha()
        else:
            continue
        # print("check captcha")
    # print()
        # If you have limited computer resources, consider adding a sleep delay between detections.
    # sleep(0.2)

    # print('Finished.')
        


def main(*argv):
    handle = win32gui.FindWindow(None, window_name)
    print("Window `{0:s}` handle: 0x{1:016X}".format(window_name, handle))
    if not handle:
        print("Invalid window handle")
        return
    remote_thread, _ = wproc.GetWindowThreadProcessId(handle)
    wproc.AttachThreadInput(wapi.GetCurrentThreadId(), remote_thread, True)
    # prev_handle = win32gui.SetForegroundWindow(handle)
    prev_handle = win32gui.SetFocus(handle)


# Function to simulate hitting the monster
def auto_hit():
    skills()  # Replace 'f' with the key used for hitting

def jump_unjump_character(coordinate):
    if coordinate['y'] > 350 and coordinate['y'] <= 500:
        # right()

        unjump()
        # skills()
        # skills()

        # i+=1
    if coordinate['y'] < 300 and coordinate['y'] >= 50: 
        # left()

        jump()
        # skills()

def automate():
    screen_height = 1080
    border_pixels = 8
    titlebar_pixels = 33
    velocity_multiplyer = 2

    previous_coordinates = []
    new_coordinates = []

    sleep(4)
    
    while(True):
        cv.destroyAllWindows()
        ss = wincap.get_screenshot()
        previous_coordinates = improc.proccess_image(ss)
        previous_coordinates = [c for c in previous_coordinates if c["class"] == np.int64(1)]
        
        ss = wincap.get_screenshot()
        new_coordinates = improc.proccess_image(ss)
        new_coordinates = [c for c in new_coordinates if c["class"] == np.int64(1)]

        # if cv.waitKey(1) == ord('q'):
        #     cv.destroyAllWindows()
        #     break
        
        if len(previous_coordinates) == 0 or len(new_coordinates) == 0:
            continue
            
        coordinates_to_hit = []

        for new_target_monster in new_coordinates:
            center_x = new_target_monster['x'] + (new_target_monster['w'] // 2) + border_pixels
            center_y = new_target_monster['y'] + (new_target_monster['h'] // 2) + titlebar_pixels
            for previous_target_monster in previous_coordinates:
                if not previous_target_monster['x'] < center_x < (previous_target_monster['x'] + previous_target_monster['w']):
                    continue
                if not previous_target_monster['y'] < center_y < (previous_target_monster['y'] + previous_target_monster['h']):
                    continue
                previous_center_x = previous_target_monster['x'] + (previous_target_monster['w'] // 2) + border_pixels
                previous_center_y = previous_target_monster['y'] + (previous_target_monster['h'] // 2) + titlebar_pixels
                coordinates_to_hit.append({
                    "x": center_x + (center_x - previous_center_x) * velocity_multiplyer,
                    "y": center_y + (center_y - previous_center_y) * velocity_multiplyer
                })
                break
        
        if len(coordinates_to_hit) == 0:
            continue
        
        if len(coordinates_to_hit) == 1:
            coordinates_to_hit = coordinates_to_hit[0]

            initial_x = coordinates_to_hit["x"]
            initial_y = min(screen_height, coordinates_to_hit["y"]) - 70
            movement_x = 0
            movement_y = 140
        
        else:
            leftmost_target_monster = min(coordinates_to_hit, key=lambda x: x['x'])
            rightmost_target_monster = max(coordinates_to_hit, key=lambda x: x['x'])
            
            initial_x = max(30, leftmost_target_monster["x"])
            initial_y = max(100, rightmost_target_monster["y"])
            movement_x = rightmost_target_monster["x"] - initial_x
            movement_y = rightmost_target_monster["y"] - initial_y
        
        print("movement x : ", movement_x)
        print("movement y : ", movement_y)
        # focusing_on()

        
        
        
        while True:
            pydirectinput.PAUSE = 0.015
            
            if initial_y > movement_y:
                unjump()
            if initial_y < movement_y:
                jump()
            if movement_x == 0:
                jump()
                skills()
                
                
                right()
            if movement_y == 0:
                
                skills()
                # right()
            if movement_x > 300:
                # continue
                skills()
                right()
                
                
                
            if movement_x < 300:
                left()
                skills()
                
                
            skills()
            unjump()
            # right()
            
            pydirectinput.press("pagedown")
            pydirectinput.press("`")
            pydirectinput.press("up")
            asyncio.run(scanning_captcha())


    # print('Finished.')


foo = Regulator(0.5, solve_captcha)
foobar = Regulator(0.5, if_captcha)

async def scanning_captcha():   
    await asyncio.gather(foobar(), foo())
    while True:
        captcha = await if_captcha()
        # await asyncio.sleep(1.0)
        if captcha == True:
            pydirectinput.press("1")
            # await asyncio.sleep(1.0)  
            await solve_captcha()
            pydirectinput.press("1")
        else:
            pydirectinput.press("3")
            break

if __name__ == "__main__":
    
    print("Python {0:s} {1:d}bit on {2:s}\n".format(" ".join(item.strip() for item in sys.version.split("\n")), 64 if sys.maxsize > 0x100000000 else 32, sys.platform))
    main(*sys.argv[1:])
    # work()
    automate()
    
    # while True:
    #     skills()
    
    
    print("\nDone.")
