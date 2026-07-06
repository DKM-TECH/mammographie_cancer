from fastapi import APIRouter, Header, HTTPException
from auth import verify_token

router = APIRouter()

# MOCK DATA (remplace par DB plus tard)
patients = [
    {"id": 1}, {"id": 2}, {"id": 3}
]

diagnostics = [
    {"id": 1, "result": "Suspect"},
    {"id": 2, "result": "Normal"},
    {"id": 3, "result": "Suspect"},
]

@router.get("/dashboard")
def get_dashboard(authorization: str = Header(None)):

    if not authorization:
        raise HTTPException(status_code=401, detail="No token")

    token = authorization.split(" ")[1]
    user = verify_token(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    suspects = len([d for d in diagnostics if d["result"] == "Suspect"])

    return {
        "patients": len(patients),
        "diagnostics": len(diagnostics),
        "suspects": suspects,
        "doctor": {
            "name": "Dr. Yvette KANKU"
        },
        "chart": {
            "labels": ["Lun", "Mar", "Mer", "Jeu", "Ven"],
            "values": [5, 10, 7, 12, 9],
            "pie": [60, 25, 15]
        }
    }