from tkinter import *
from PIL import ImageTk, Image
import pollution_prediction
import temperature_prediction

root = Tk()

w = Label(root, text="Weather conditions ", fg="black", font=("Helvetica", 20))
w1 = Label(root, text="Suggested activities :", fg="black", font=("Helvetica", 20))

w.pack()
w1.pack()
canvas = Canvas(root, width=1124, height=720)
canvas.pack()

temp_low = 18
temp_mid = 27
pm2_limit = 60
temp = temperature_prediction.export
pm2 = pollution_prediction.export
if (temp > temp_mid and pm2 > pm2_limit):
    img1 = ImageTk.PhotoImage(Image.open("hot_temp.jpg"))
    img2 = ImageTk.PhotoImage(Image.open("high_pollution.jpg"))
    canvas.create_image(50, 0, anchor=NW, image=img1)
    canvas.create_image(510, 0, anchor=NW, image=img2)
if (temp > temp_mid and pm2 < pm2_limit):
    img1 = ImageTk.PhotoImage(Image.open("hot_temp.jpg"))
    img2 = ImageTk.PhotoImage(Image.open("safe_pollution.jpg"))
    canvas.create_image(90, 0, anchor=NW, image=img1)
    canvas.create_image(560, 0, anchor=NW, image=img2)
if (temp <= temp_mid and temp >= temp_low and pm2 > pm2_limit):
    img1 = ImageTk.PhotoImage(Image.open("mod_temp.jpg"))
    img2 = ImageTk.PhotoImage(Image.open("high_pollution.jpg"))
    canvas.create_image(40, 0, anchor=NW, image=img1)
    canvas.create_image(530, 0, anchor=NW, image=img2)
if (temp <= temp_mid and temp >= temp_low and pm2 < pm2_limit):
    img1 = ImageTk.PhotoImage(Image.open("mod_temp.jpg"))
    img2 = ImageTk.PhotoImage(Image.open("safe_pollution.jpg"))
    canvas.create_image(70, 0, anchor=NW, image=img1)
    canvas.create_image(570, 0, anchor=NW, image=img2)
if (temp < temp_low and pm2 > pm2_limit):
    img1 = ImageTk.PhotoImage(Image.open("cool_temp.jpg"))
    img2 = ImageTk.PhotoImage(Image.open("high_pollution.jpg"))
    canvas.create_image(70, 0, anchor=NW, image=img1)
    canvas.create_image(480, 0, anchor=NW, image=img2)
if (temp < temp_low and pm2 < pm2_limit):
    img1 = ImageTk.PhotoImage(Image.open("cool_temp.jpg"))
    img2 = ImageTk.PhotoImage(Image.open("safe_pollution.jpg"))
    canvas.create_image(70, 0, anchor=NW, image=img1)
    canvas.create_image(570, 0, anchor=NW, image=img2)
root.mainloop()
