import tkinter as tk
from tkinter import *

def create_window(ImageZoneWidth, ImageZoneHeight, ColonneWidth, NumLignes):

    window = tk.Tk()

    window.title("Image Detection")  # set the title of the window  
    taille = str(ImageZoneWidth+ColonneWidth)+'x'+str(ImageZoneHeight)
    window.geometry(taille)  # set the size of the window        

    #move window to center of the screen
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (ImageZoneWidth // 2)
    y = (screen_height // 2) - (ImageZoneHeight // 2)
    window.geometry('+{}+{}'.format(x, y))  

#    def on_closing():
#        if messagebox.askokcancel("Quit", "Do you want to quit?"):
#            window.quit()  
#            window.destroy()
#            exit

#    window.protocol("WM_DELETE_WINDOW", on_closing)

    #manage the menu bar
    menubar = tk.Menu(window)
    
    file_menu = Menu(menubar, tearoff=0)
    file_menu.add_command(label='Quit', command=window.destroy)
    menubar.add_cascade(label='File', menu=file_menu)
    window.config(menu=menubar)


    #create a grid layout for the window
    window.columnconfigure(0, minsize=ColonneWidth,)
    window.columnconfigure(1, minsize=ImageZoneWidth)
    for i in range(NumLignes):
        window.rowconfigure(i, weight=1)

        
    # create a canvas to display the image
    canvas = tk.Canvas(window, width=ImageZoneWidth, height=ImageZoneHeight)
    canvas.grid (row=1, column=1, sticky="nsew", rowspan=NumLignes-1)
    #set canvas background color
    canvas.config(background="#E0E0E0")
    canvas.image = None



    return window, canvas, menubar
