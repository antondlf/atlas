import tkinter as tk
from tkinter.filedialog import askopenfilename
from pathlib import Path
from asr import main

window = tk.Tk()

window.title("Automatic Transcription for Linguistic Annotators")

def show():
    filename = askopenfilename(filetypes=[("Wav files", '*.wav')])
    main(filename)
    print(filename)
    #main(path)

button = tk.Button(window, text='Transcribe a file', width=25, command=lambda: show())
button.place(x=50, y=50)
window.mainloop()