from fastapi import FastAPI, File
from nsfw_detector import predict
from random import randint
import os
import uvicorn

model = predict.load_model('./saved_model.h5')

app = FastAPI()

@app.post("/is_nsfw")
def read_root(name: str = None, file: bytes = File(default=None)):
    if not file:
        return {"status_code": "400", "status": "File upload failed"}
    
    if not name:
        return {"status_code": "400", "status": "Need full file name"}
    
    name = f"{randint(0,100000)}{os.path.splitext(name)[1]}"
    with open(name, "wb") as f:
        f.write(file)
    
    answer = predict.classify(model, name)
    os.remove(name)
    return {"status_code": "200", "status": "succesfull", "answer": answer[name]}

if __name__ == "__main__":
    uvicorn.run('main:app',
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 8080)),
        log_level="debug",
        http="h11",
        use_colors=True,
        workers=3
    )
