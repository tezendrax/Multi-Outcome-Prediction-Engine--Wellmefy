import uvicorn
from mope.config import settings

if __name__ == "__main__":
    print(f"Starting Multi-Outcome Prediction Engine (MOPE) on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "mope.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
