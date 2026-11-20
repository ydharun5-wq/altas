"""Add database-backed login throttling.

Revision ID: 0002_login_attempts
Revises: 0001_initial
"""
from alembic import op
import sqlalchemy as sa


revision = "0002_login_attempts"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "login_attempts",
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key_hash"),
    )
    op.create_index("ix_login_attempts_updated", "login_attempts", ["updated_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_login_attempts_updated", table_name="login_attempts")
    op.drop_table("login_attempts")
