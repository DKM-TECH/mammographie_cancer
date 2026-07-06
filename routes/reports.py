from fastapi import APIRouter

router = APIRouter()

@router.get("/report")
def get_report():
    return {
        "status": "OK",
        "message": "Report generated successfully"
    }