from pydantic import BaseModel


class Column(BaseModel):
    field: str
    headerText: str
    visible: bool


class ReportPayload(BaseModel):
    model: str
    language: str
    count: int
    cloumns: list[Column]
    dataSrcList: list[dict]
