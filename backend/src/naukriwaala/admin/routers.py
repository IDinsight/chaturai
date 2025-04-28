"""This module contains FastAPI routers for admin endpoints."""

# Third Party Library
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

# Package Library
from naukriwaala.db.utils import get_async_session

# Globals.
TAG_METADATA = {
    "description": "Application healthcheck.",
    "name": "Admin",
}

router = APIRouter(tags=[TAG_METADATA["name"]])


@router.get("/health")
async def healthcheck(
    asession: AsyncSession = Depends(get_async_session),
) -> JSONResponse:
    """Healthcheck endpoint - checks connection to the database.

    Parameters
    ----------
    asession
        The SQLAlchemy async session to use for all database connections.

    Returns
    -------
    JSONResponse
        A JSON response with the status of the database connection.
    """

    try:
        await asession.execute(text("SELECT 1;"))
    except SQLAlchemyError as e:
        return JSONResponse(
            content={"message": f"Failed database connection: {e}"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return JSONResponse(content={"status": "ok"}, status_code=status.HTTP_200_OK)
