import os
from tkinter import filedialog
import tkinter as tk


def ask_filenames(ftypes=[("", "*.")]):
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_names = filedialog.askopenfilenames(filetypes = ftypes,initialdir = iDir)
    dir_name   = os.path.dirname(file_names[0])
    return dir_name, file_names

def ask_directory(init_dir=None):
    root = tk.Tk()
    root.withdraw()
    if not init_dir:
        init_dir = os.path.abspath(os.path.dirname(__file__))
    dir_name = filedialog.askdirectory(initialdir = init_dir)
    return dir_name

def input_is_yes(print_word):
    while True:
        y_or_n = input(print_word + "(y/n)\n>>")
        if y_or_n == "y" or y_or_n == "n":
            break   
    return y_or_n == 'y'

if __name__ == '__main__':
    ask_filenames()
    ask_directory()