from fastapi import APIRouter, HTTPException
from auth import create_token

router = APIRouter()

fake_user = {
    "email": "doctor@onco.ai",
    "password": "1234",
    "name": "Dr. Kabongo"
}

@router.post("/login")
def login(data: dict):
    if data["email"] == fake_user["email"] and data["password"] == fake_user["password"]:
        token = create_token({"sub": fake_user["email"]})
        return {
            "access_token": token,
            "doctor": fake_user
        }

    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/logout")
def logout():
    return {"message": "Logged out"}