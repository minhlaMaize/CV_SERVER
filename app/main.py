from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .routers import predict
app = FastAPI()

app.include_router(predict.router)

@app.get("/")
async def main():
    content = """
        <body>
        <form action="/predict/" enctype="multipart/form-data" method="post">
        <input name="file" type="file" multiple>
        <input type="submit">
        </body>
            """
    return HTMLResponse(content=content)