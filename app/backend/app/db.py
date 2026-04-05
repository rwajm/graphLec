import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        # Create table if not exists
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS jobs (
                id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                status           TEXT NOT NULL DEFAULT 'pending',
                input_path       TEXT NOT NULL,
                output_dir       TEXT,
                current_stage    TEXT,
                error_message    TEXT,
                pipeline_stages  JSONB,
                created_at       TIMESTAMPTZ DEFAULT now(),
                updated_at       TIMESTAMPTZ DEFAULT now()
            );
        """))
        
        # Check if 'status' column exists (for backward compatibility if table existed)
        # We can try to add it, if it fails it's okay (already exists)
        try:
            await conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'pending'"))
            await conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS current_stage TEXT"))
            await conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS error_message TEXT"))
            await conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS pipeline_stages JSONB"))
        except Exception:
            pass # Ignore if already exists or other error

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
