from fastapi import APIRouter

router = APIRouter()

patients = []

@router.post("/patients")
def add_patient(patient: dict):
    patients.append(patient)
    return patient

@router.get("/patients")
def get_patients():
    return patients

@router.put("/patients/{id}")
def update_patient(id: int, data: dict):
    for p in patients:
        if p.get("id") == id:
            p.update(data)
            return p
    return {"error": "Not found"}

@router.delete("/patients/{id}")
def delete_patient(id: int):
    global patients
    patients = [p for p in patients if p.get("id") != id]
    return {"message": "Deleted"}