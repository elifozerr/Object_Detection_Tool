import shutil
import uuid
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List,Optional
from image_detect.detect_image import Detector
import pathlib
from fastapi.responses import Response, FileResponse, PlainTextResponse, HTMLResponse

app = FastAPI()
IMAGEDIR =  str(pathlib.Path().absolute()) + "/image_detect/images/" 

app.mount("/image_detect/images", StaticFiles(directory="image_detect/images"), name="image_detect/images")

@app.get("/")
def read_root():
    return {"greetings": "Welcome to JotForm"}



@app.post("/img")
async def upload_image(file: UploadFile = File(...)):
   
    file.filename = f"{uuid.uuid4()}.jpg"

    contents = await file.read()  # <-- Important!
    path = f"{IMAGEDIR}{file.filename}"
    print("test" +path)
    print("test2")
    print( str(pathlib.Path().absolute()))
    with open(path, "wb") as f:
        f.write(contents)


    detector = Detector(path)
    label = detector.detect()


    #return {"object": label}
    return display_result(label,path,file.filename)

def display_result(label, path, image_file):

    print(str(image_file))
    html_content = \
    '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <style>
    * {
        box-sizing: border-box;
        -moz-box-sizing: border-box;
        -webkit-box-sizing: border-box;
      }

      body {
        font-family: 'Montserrat', sans-serif;
        background: #fa8900;
      }
      .wrapper {
        margin: auto;
        max-width: 640px;
        padding-top: 60px;
        text-align: center;
      }

      .container {
        background-color: #fa8900;
        padding: 20px;
        border-radius: 10px;
        
      }

      h1 {
        color: #292626;
        font-family: 'Varela Round', sans-serif;
        letter-spacing: -.5px;
        font-weight: 700;
        padding-bottom: 10px;
      }
    </style> 
    </head>
    <body>
        <div class="wrapper">
            <div class="container">
            <h2>Original Image</h2>

            <img src= "/image_detect/images/''' + image_file + '''" style="max-width: 200px;">'

            
            
            <p style="margin-bottom: 20px;">Detected Object :  ''' + label + '''</p>
            
            

            <img class="logo" src="https://www.jotform.com/resources/assets/logo/new/400x100/jotform-logo-transparent-400x100.png"></img>
            </div>
        </div>
        
    </body>
    </html>
    '''
    
    return HTMLResponse(content = html_content, status_code=200, media_type='text/html')
