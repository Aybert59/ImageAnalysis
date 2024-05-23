import tkinter as tk
from tkinter import *
from tkinter import messagebox

TextArea = None

def display_log(text,category):
    global TextArea

    # category : info, warning, error
    TextArea.config(state="normal")
    if (category == "info"):
        #insert text in the text area  
        TextArea.insert(tk.END, text + "\n")
    elif (category == "warning"):
        #insert tert in bold in the text area
        TextArea.tag_add("bold", "1.0", "end")
        TextArea.tag_config("bold", font=("Helvetica", 12, "bold"))
        TextArea.insert(tk.END, text + "\n")
    elif (category == "error"):
        #insert text in red and bold in the text area
        TextArea.tag_add("bold", "1.0", "end")
        TextArea.tag_config("bold", font=("Helvetica", 12, "bold"))
        TextArea.tag_add("bold", "1.0", "end")
        TextArea.tag_config("bold", foreground="red")
        TextArea.insert(tk.END, text + "\n")

    TextArea.see(tk.END)
    TextArea.config(state="disabled")

def create_window(ImageZoneWidth, ImageZoneHeight, ColonneWidth):
    global TextArea
    
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

    def on_closing():
       if messagebox.askokcancel("Quit", "Do you want to quit?"):
            window.quit()  
            window.destroy()
            exit

    window.protocol("WM_DELETE_WINDOW", on_closing)

    #manage the menu bar
    menubar = tk.Menu(window)
    
    file_menu = Menu(menubar, tearoff=0)
    #file_menu.add_command(label='Quit', command=window.destroy)
    file_menu.add_command(label='Quit', command=on_closing)
    menubar.add_cascade(label='File', menu=file_menu)
    window.config(menu=menubar)


    # create a canvas to display the image
    canvas = tk.Canvas(window, width=ImageZoneWidth, height=ImageZoneHeight, bg="#E0E0E0")
    canvas.image = None
    canvas.pack(side=tk.RIGHT)

    widget_bar = Frame(window)
    widget_bar.pack(side=tk.RIGHT, fill=tk.BOTH, pady=20, expand=YES)
    widget_bar.config(background="#C0C0C0")

    TextArea = Text(widget_bar, height=15, width=20)
    TextArea.pack(side=tk.BOTTOM, fill=tk.X)
    TextArea.config(state='disabled')

    TextArea.pack(side=tk.BOTTOM, fill=tk.X)
    tk.Label(widget_bar, text="Information:        ").pack(side=tk.BOTTOM, fill=tk.X)
    tk.Label(widget_bar, text="").pack(ipady=30, side=tk.BOTTOM, fill=tk.X) # space
    

    return window, canvas, widget_bar, menubar
