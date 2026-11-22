"""Initial historical forecast run using the existing LangGraph/MCP workflow.

Runs a bounded batch of real products. Does not modify the source dataset.
"""
import asyncio
import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from google.cloud import storage

BUCKET = os.getenv('PHARMA_GCS_BUCKET', 'pharma-auto-forecasting-data')
DATA_PATH = '/tmp/pharma/data/nhs_monthly_prescribing.csv'
os.environ['PHARMA_DATA_PATH'] = DATA_PATH  # Set BEFORE project imports.

from orchestrator.workflow import run_workflow
from tools.cloud_persistence import restore_state, save_state
from tools.cloud_hitl import sync_reviews_for_forecasting


def eligible_codes(path):
    frame = pd.read_csv(path, dtype={'bnf_chemical_substance_code': str})
    required = {'month', 'bnf_chemical_substance_code', 'items'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing dataset columns: {sorted(required - set(frame.columns))}')
    frame['month'] = pd.to_datetime(frame['month'], format='%Y-%m', errors='coerce')
    frame['items'] = pd.to_numeric(frame['items'], errors='coerce')
    frame = frame.dropna(subset=['month', 'bnf_chemical_substance_code', 'items'])
    frame = frame[frame['items'] >= 0]
    counts = frame.groupby('bnf_chemical_substance_code')['month'].nunique()
    return sorted(counts[counts >= 36].index.tolist())


def json_safe(value):
    """Convert numpy/pandas dates and numeric values to JSON-compatible data."""
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def persist_forecast(bucket, code, state, forecast, evaluation):
    """Store the actual workflow forecast payload in a separate object per medicine."""
    if forecast.get('status') != 'SUCCESS':
        return None
    payload = {
        'product_code': code,
        'workflow_status': state.get('status'),
        'champion_model': evaluation.get('champion_model'),
        'forecast_result': forecast,
    }
    object_name = f'forecasts/{code}.json'
    bucket.blob(object_name).upload_from_string(
        json.dumps(payload, default=json_safe, allow_nan=False, indent=2),
        content_type='application/json',
    )
    print(f'Saved forecast: gs://{BUCKET}/{object_name}', flush=True)
    return object_name


async def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    bucket.blob('raw/nhs_monthly_prescribing.csv').download_to_filename(DATA_PATH)
    print('Historical NHS dataset downloaded', flush=True)
    restore_state()
    review_count = sync_reviews_for_forecasting()
    print(f"HITL review decisions loaded: {review_count}", flush=True)

    codes = eligible_codes(DATA_PATH)
    offset = int(os.getenv('PHARMA_BATCH_OFFSET', '0'))
    limit = int(os.getenv('PHARMA_BATCH_LIMIT', '3'))
    if offset < 0 or limit < 1:
        raise ValueError('Batch offset must be >= 0 and batch limit >= 1')
    selected = codes[offset:offset + limit]
    print(f'Eligible medicines: {len(codes)}; processing {len(selected)} from offset {offset}', flush=True)

    outcomes = []
    for code in selected:
        try:
            # Skip medicines with a previously saved successful forecast.
            forecast_blob = bucket.blob(f'forecasts/{code}.json')
            if forecast_blob.exists():
                existing = json.loads(forecast_blob.download_as_text())
                saved_forecast = existing.get('forecast_result') or {}
                if (existing.get('product_code') == code
                        and existing.get('workflow_status') == 'SUCCESS'
                        and saved_forecast.get('status') == 'SUCCESS'
                        and len(saved_forecast.get('forecast') or {}) == 12):
                    outcomes.append({
                        'product_code': code,
                        'status': 'SKIPPED_EXISTING_FORECAST',
                        'forecast_generated': True,
                        'forecast_saved': True,
                    })
                    print(json.dumps(outcomes[-1]), flush=True)
                    continue
            state = await run_workflow(
                f'Forecast medicine {code} for the next 12 months',
                user_id='cloud_historical_system',
            )
            forecast = state.get('forecast_result') or {}
            evaluation = state.get('evaluation_result') or {}
            saved_object = persist_forecast(bucket, code, state, forecast, evaluation)
            outcomes.append({
                'product_code': code,
                'status': state.get('status'),
                'champion_model': evaluation.get('champion_model'),
                'forecast_generated': forecast.get('status') == 'SUCCESS',
                'forecast_saved': saved_object is not None,
            })
        except Exception as exc:
            outcomes.append({'product_code': code, 'status': 'FAILED', 'error': str(exc)})
        finally:
            # Keep durable outputs after each medicine, even if a later one fails.
            save_state()
        print(json.dumps(outcomes[-1]), flush=True)

    print(json.dumps({'eligible_count': len(codes), 'batch_offset': offset,
                      'processed': len(outcomes), 'results': outcomes}, indent=2), flush=True)
    # Individual failures/reviews are reported above; do not fail the whole batch.
    print('Historical batch finished; inspect per-medicine statuses above.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
