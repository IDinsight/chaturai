"""add_timezone

Revision ID: 29480d9f3e71
Revises: 2d58750c03da
Create Date: 2025-04-17 17:11:06.871605

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "29480d9f3e71"  # pragma: allowlist secret
down_revision: Union[str, None] = "2d58750c03da"  # pragma: allowlist secret
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """alter the column to be timestamp with timezone"""
    op.alter_column(
        "opportunities",
        "last_checked",
        type_=sa.DateTime(timezone=True),
        postgresql_using="last_checked AT TIME ZONE 'Asia/Kolkata'",
        existing_type=sa.DateTime(),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Remove timezone awareness"""
    op.alter_column(
        "opportunities",
        "last_checked",
        type_=sa.DateTime(),
        postgresql_using="last_checked AT TIME ZONE 'Asia/Kolkata'",
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=True,
    )
