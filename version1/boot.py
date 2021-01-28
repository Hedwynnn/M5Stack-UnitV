from modules import ws2812
from fpioa_manager import *
import sensor
import image
import lcd
import time
import KPU as kpu
import gc
import sys
from fpioa_manager import fm
from machine import I2C
from Maix import I2S, GPIO
import time
import uos
import os
#RGB
fm.register(8)
class_ws2812 = ws2812(8, 100)
#RIGHT BUTTON
fm.register(18, fm.fpioa.GPIO1)
but_a=GPIO(GPIO.GPIO1, GPIO.IN, GPIO.PULL_UP) #PULL_UP
#LEFT BUTTON
fm.register(19, fm.fpioa.GPIO2)
but_b = GPIO(GPIO.GPIO2, GPIO.IN, GPIO.PULL_UP) #PULL_UP

def RGB_GREEN():
    a = class_ws2812.set_led(0, (0,100, 0))
    a = class_ws2812.display()
    time.sleep(0.5)
    a = class_ws2812.set_led(0, (0, 0, 0))
    a = class_ws2812.display()

def RGB_RED():
    a = class_ws2812.set_led(0, (100,0, 0))
    a = class_ws2812.display()
    time.sleep(0.5)
    a = class_ws2812.set_led(0, (0, 0, 0))
    a = class_ws2812.display()

def RGB_BLUE():
    a = class_ws2812.set_led(0, (0,0, 100))
    a = class_ws2812.display()
    time.sleep(0.5)
    a = class_ws2812.set_led(0, (0, 0, 0))
    a = class_ws2812.display()
#SENSOR初期化
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224,224))
sensor.set_hmirror(1)
sensor.set_vflip(1)
sensor.run(1)

position_x = 0
position_y = 0
aru = 0
count = 1
currentImage = 0

task = kpu.load("/sd/m.kmodel") #YOLO.v2
labels = ["clock"]
anchors = [1.3125, 1.09375, 2.28125, 1.71875, 2.1875, 2.03125, 0.6875, 0.65625, 2.4375, 2.21875]

try:
    os.mkdir("/sd/photo")
except Exception as e:
    pass

while(True):
    if but_a.value() == 0:
        RGB_RED()
        clock = time.clock()
        start_time = time.ticks_ms()
        try:
            kpu.init_yolo2(task, 0.2, 0.5, 5, anchors)
            while(True):
                    if but_b.value() == 0:
                        RGB_BLUE()
                        break
                    img = sensor.snapshot()
                    t = time.ticks_ms()
                    objects = kpu.run_yolo2(task, img)
                    t = time.ticks_ms() - start_time
                    img.draw_string(0,0, "count:%d" %(count), scale=2, color=(255, 255, 255))
                    if objects:
                        RGB_GREEN()
                        img.draw_string(170,0, "aru:%d" %(aru), scale=2, color=(255, 255, 255))
                        for obj in objects:
                            pos = obj.rect()
                            img.draw_rectangle(pos)
                            img.draw_string(pos[0], pos[1], "%s : %.2f" %(labels[obj.classid()], obj.value()), scale=2, color=(255, 0, 0))
                            position_x = obj.rect()[0] + obj.rect()[2]//2
                            position_y = obj.rect()[1] + obj.rect()[3]
                            #print(position_x,position_y)
                            img.draw_string(0, 200, "t:%ds" %(t/1000), scale=2, color=(0, 0, 255))
                            #lcd.display(img)
                            if(aru == 0 and 20<position_x<180 and 20<position_y<180):
                                photo = img.save("/sd/photo/"+ str(currentImage) + ".jpg", quality=95)
                                currentImage = currentImage + 1
                                with open("/sd/result.txt","a") as f:
                                    f.write("clock : %d : %ds; \n" %(count,t/1000))
                                count += 1
                            aru += 1
                    else:
                        aru = 0
                        img.draw_string(0, 200, "t:%dms" %(t), scale=2, color=(255, 255, 255))
        finally:
            gc.collect()
a = kpu.deinit(task)
