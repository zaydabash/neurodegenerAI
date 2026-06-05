"""Initial schema: patients, neuro_predictions, social_posts, trend_topics

Revision ID: 0001
Revises:
Create Date: 2025-12-19

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "patients",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("external_id", sa.String(length=50), nullable=True),
        sa.Column("age", sa.Float(), nullable=True),
        sa.Column("sex", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_patients_id", "patients", ["id"])
    op.create_index("ix_patients_external_id", "patients", ["external_id"], unique=True)

    op.create_table(
        "neuro_predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=True),
        sa.Column("model_type", sa.String(length=50), nullable=True),
        sa.Column("prediction", sa.Integer(), nullable=True),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("results_metadata", sa.PickleType(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_neuro_predictions_id", "neuro_predictions", ["id"])

    op.create_table(
        "social_posts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=True),
        sa.Column("url", sa.String(length=255), nullable=True),
        sa.Column("author", sa.String(length=100), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_social_posts_id", "social_posts", ["id"])

    op.create_table(
        "trend_topics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("topic", sa.String(length=255), nullable=False),
        sa.Column("keywords", sa.PickleType(), nullable=True),
        sa.Column("trending_score", sa.Float(), nullable=True),
        sa.Column("volume", sa.Integer(), nullable=True),
        sa.Column("growth_rate", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trend_topics_id", "trend_topics", ["id"])


def downgrade() -> None:
    op.drop_index("ix_trend_topics_id", table_name="trend_topics")
    op.drop_table("trend_topics")
    op.drop_index("ix_social_posts_id", table_name="social_posts")
    op.drop_table("social_posts")
    op.drop_index("ix_neuro_predictions_id", table_name="neuro_predictions")
    op.drop_table("neuro_predictions")
    op.drop_index("ix_patients_external_id", table_name="patients")
    op.drop_index("ix_patients_id", table_name="patients")
    op.drop_table("patients")
