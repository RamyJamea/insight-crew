from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from ...core import create_llm
from .schemas import ReportPayload
from .utils import convert_to_df
from .workflow import run_workflow

router = APIRouter(prefix="/nlp", tags=["Reporting"])


@router.post("/insights")
def generate_insights(payload: ReportPayload):
    try:
        target_dataframe = convert_to_df(payload.model_dump())
        llm = create_llm(payload.model_provider, payload.model_code, payload.api_key)
        final_report = run_workflow(target_dataframe, llm, payload.language)

        return JSONResponse(
            content={
                "rows_processed": target_dataframe.shape[0],
                "report": final_report.model_dump(),
            },
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
