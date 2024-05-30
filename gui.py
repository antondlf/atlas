import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from pathlib import Path
from asr import transcribe_audio


class windows(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.arg_dict = {
            'file_path': None,
            'n_speakers': None,
            'language': None,
            'output_path': None,
            'output_app': 'Elan'
        }
        # Adding a title to the window
        self.wm_title("Automatic Transcription for Linguistic Annotations")

        # creating a frame and assigning it to container
        container = tk.Frame(self, bg='black')
        # specifying the region where the frame is packed in root
        container.pack(side="top", fill="both", expand=True)

        # configuring the location of the container using grid
        container.grid_rowconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)
        container.grid_rowconfigure(2, weight=1)
        container.grid_rowconfigure(3, weight=1)
        container.grid_rowconfigure(4, weight=1)
        container.grid_rowconfigure(5, weight=1)
        container.grid_rowconfigure(6, weight=1)
        container.grid_rowconfigure(7, weight=1)
        container.grid_rowconfigure(8, weight=1)
        container.grid_rowconfigure(9, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # We will now create a dictionary of frames
        self.frames = {}
        # we'll create the frames themselves later but let's add the components to the dictionary.
        for F in (MainPage, SidePage, CompletionScreen):
            frame = F(container, self)

            # the windows class acts as the root window for the frames.
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Using a method to switch frames
        self.show_frame(MainPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        # raises the current frame to the top
        frame.tkraise()


class MainPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.arg_dict = controller.arg_dict
        #self.grid(row=0, column=0)
        main_frame = tk.Frame(self, bg='black')
        main_frame.tkraise()
        label = tk.Label(main_frame, text="Main Page")
        

        # Language selection
        language_selector_title = tk.Label(main_frame, text='Select a Language')
        self.languages_supported = ["Galician", "English"]
        choicesvar = tk.StringVar(value=self.languages_supported)
        self.language_selector = tk.Listbox(
            main_frame,
            listvariable=choicesvar,
            height=10,
            width = 15,
            activestyle = 'dotbox', 
            font = "Helvetica",
            fg = "red"
        )
        
        #self.language_selector.bind('<<ListboxSelect>>', self.set_language)


        # Speaker number selection
        n_speaker_title = tk.Label(main_frame, text="How many speakers do you expect in this audio.")

        self.n_speaker_box = tk.Entry(main_frame, width=25, bd =5, bg='white')
        
        #self.n_speaker_box.set('2')

        # Output File path
        output_path_title = tk.Label(main_frame, text="Output filepath for transcription.")
        
        self.output_path = tk.Entry(main_frame, width=25, bd =5, bg="white")
        
        self.button = tk.Button(main_frame, text='Select file for transcription', width=25, command=lambda: self.get_file())
        
        # We use the switch_window_button in order to call the show_frame() method as a lambda function
        self.switch_window_button = tk.Button(
            main_frame,
            text="Transcribe",
            command=lambda: self.fill_args(controller),
        )
        main_frame.grid(row=0, column=0,  sticky='nsew')
        label.grid(row=0, column=0,   sticky='nsew')
        language_selector_title.grid(row=1, column=0,  sticky='nsew')
        self.language_selector.grid(row=2,column=0, rowspan=2, columnspan=2,  sticky='nsew')

        n_speaker_title.grid(row=4, column=0,  sticky='nsew')
        self.n_speaker_box.grid(row=5, column=0,  sticky='nsew')
        
        output_path_title.grid(row=6, column=0, sticky='nsew')
        self.output_path.grid(row=7, column=0, sticky='nsew')
        
        self.button.grid(row=8, column=0,  sticky='nsew')
        self.switch_window_button.grid(row=9, column=0,  sticky='nsew')
        

    def get_file(self):
            print(self.output_path.get())
            self.arg_dict['file_path'] = askopenfilename(filetypes=[("Wav files", '*.wav')])

    def set_language(self):
        idx = self.language_selector.curselection()
        self.arg_dict['language'] = self.languages_supported[idx]


    def fill_args(self, controller):
        self.arg_dict['n_speakers']  = int(self.n_speaker_box.get())
        self.arg_dict['language']    = self.language_selector.get(self.language_selector.curselection()[0])
        self.arg_dict['output_path'] = self.output_path.get()
        self.arg_dict['output_app']  = 'Elan' # To add option when supported.
        self.arg_dict['quantization']= 'float32' # to add option when supported.


        transcribe_audio(
            self.arg_dict['file_path'],
            self.arg_dict['language'],
            self.arg_dict['output_path'],
            self.arg_dict['quantization'],
            self.arg_dict['n_speakers']
            )
        controller.show_frame(SidePage)
        


class SidePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Your transcription is in the output file.")
        label.pack(padx=10, pady=10)

        switch_window_button = tk.Button(
            self,
            text=f"Processing audio...",
            command=lambda: controller.show_frame(CompletionScreen),
        )
        switch_window_button.pack(side="bottom", fill=tk.X)


class CompletionScreen(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Transcription done!")
        label.pack(padx=10, pady=10)
        switch_window_button = tk.Button(
            self, text="Return to menu", command=lambda: controller.show_frame(MainPage)
        )
        switch_window_button.pack(side="bottom", fill=tk.X)


if __name__ == '__main__':
    app = windows()
    app.mainloop()
