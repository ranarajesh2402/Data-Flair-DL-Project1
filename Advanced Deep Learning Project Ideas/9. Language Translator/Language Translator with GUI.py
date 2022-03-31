#If 'NoneType' error happen try the steps below. Then Restart Kernel and run the Program.

#!pip uninstall googletrans -y         #To uninstall current version of googletrans
#!pip install googletrans==4.0.0rc1
#import googletrans
#print(googletrans.__version__)


#importing modules
from tkinter import *
from googletrans import Translator

# initializing window
Screen = Tk()
Screen.title("Language Translator with GUI by- TechVidvan")

InputLanguageChoice = StringVar()
TranslateLanguageChoice = StringVar()

#tuple for choosing languages
LanguageChoices = {'Hindi','English','French','German','Spanish'}
InputLanguageChoice.set('English')
TranslateLanguageChoice.set('Hindi')

def Translate():
    translator = Translator()
    Translation = translator.translate(TextVar.get(), src=InputLanguageChoice.get(), dest=TranslateLanguageChoice.get())
    OutputVar.set(Translation.text)

#choice for input language
InputLanguageChoiceMenu = OptionMenu(Screen,InputLanguageChoice,*LanguageChoices)
Label(Screen,text="Choose a Language").grid(row=0,column=1)
InputLanguageChoiceMenu.grid(row=1,column=1)

#choice in which the language is to be translated
NewLanguageChoiceMenu = OptionMenu(Screen,TranslateLanguageChoice,*LanguageChoices)
Label(Screen,text="Translated Language").grid(row=0,column=2)
NewLanguageChoiceMenu.grid(row=1,column=2)

Label(Screen,text="Enter Text").grid(row=2,column =0)
TextVar = StringVar()
TextBox = Entry(Screen,textvariable=TextVar).grid(row=2,column = 1)

Label(Screen,text="Output Text").grid(row=2,column =2)
OutputVar = StringVar()
TextBox = Entry(Screen,textvariable=OutputVar).grid(row=2,column = 3)

#Button for calling function
B = Button(Screen,text="Translate",command=Translate, relief = GROOVE).grid(row=3,column=1,columnspan = 3)

mainloop()

#For Reference regarding googletrans api: https://stackabuse.com/text-translation-with-google-translate-api-in-python/
#For Reference regarding Tkinter application: https://techvidvan.com/tutorials/python-language-translator/

