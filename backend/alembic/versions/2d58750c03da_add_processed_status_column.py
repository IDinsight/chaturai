"""add processed status column

Revision ID: 2d58750c03da
Revises: 39bf69aadd68
Create Date: 2024-04-16 08:19:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2d58750c03da"  # pragma: allowlist secret
down_revision: Union[str, None] = "39bf69aadd68"  # pragma: allowlist secret
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add processed_status column with default value 'open'"""
    op.add_column(
        "opportunities",
        sa.Column(
            "processed_status", sa.String(20), server_default="open", nullable=True
        ),
    )


def downgrade() -> None:
    """Remove processed_status column"""
    op.drop_column("opportunities", "processed_status")
