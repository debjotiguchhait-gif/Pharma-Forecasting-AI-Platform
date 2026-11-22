
import hashlib
import json
import os

from google.cloud import storage

from audit.logger import record_event
from orchestrator.workflow import run_workflow
from tools.cloud_monthly_ingestion import commit_cloud_monthly_data
from tools.cloud_persistence import restore_state, save_state
from tools.cloud_hitl import sync_reviews_for_forecasting


BUCKET = os.getenv(
    "PHARMA_GCS_BUCKET",
    "pharma-auto-forecasting-data",
)


def get_checkpoint_blob():
    return storage.Client().bucket(BUCKET).blob(
        "checkpoints/monthly_reassessment.json"
    )


def save_checkpoint(checkpoint):
    get_checkpoint_blob().upload_from_string(
        json.dumps(checkpoint, indent=2),
        content_type="application/json",
    )


def load_checkpoint():
    blob = get_checkpoint_blob()

    if not blob.exists():
        return None

    return json.loads(blob.download_as_text())


def get_batch_id(incoming_path):
    digest = hashlib.sha256()

    with open(incoming_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_review_fingerprint():
    """
    Fingerprint current cloud HITL decisions so that
    a changed review invalidates checkpoint reuse.
    """
    digest = hashlib.sha256()
    bucket = storage.Client().bucket(BUCKET)

    blobs = sorted(
        bucket.list_blobs(prefix="hitl/reviews/"),
        key=lambda blob: blob.name,
    )

    for blob in blobs:
        if not blob.name.endswith(".json"):
            continue

        digest.update(blob.name.encode())
        digest.update(blob.download_as_bytes())

    return digest.hexdigest()


async def run_cloud_monthly_reassessment(
    incoming_path,
    state_already_restored=False,
):
    # Restore persisted registry and audit state.
    # Set state_already_restored=True if the caller
    # has already performed this initialization.
    if not state_already_restored:
        restore_state()

    # Synchronize current cloud HITL decisions.
    review_count = sync_reviews_for_forecasting()
    print(
        f"HITL decisions synchronized: {review_count}",
        flush=True,
    )

    review_fingerprint = get_review_fingerprint()
    batch_id = get_batch_id(incoming_path)
    checkpoint = load_checkpoint()

    resumable = (
        checkpoint is not None
        and checkpoint.get("batch_id") == batch_id
        and checkpoint.get("status") in (
            "IN_PROGRESS",
            "REVIEW_REQUIRED",
        )
    )

    if resumable:
        print(
            "Resuming monthly reassessment",
            flush=True,
        )

        affected_products = checkpoint["affected_products"]

        # Previously successful forecasts are reusable
        # only if the review decisions have not changed.
        if (
            checkpoint.get("review_fingerprint")
            == review_fingerprint
        ):
            results = checkpoint.get("results", [])
        else:
            print(
                "HITL decisions changed; "
                "reassessing affected products",
                flush=True,
            )
            results = []

        checkpoint["results"] = results
        checkpoint["review_fingerprint"] = review_fingerprint
        checkpoint["status"] = "IN_PROGRESS"
        save_checkpoint(checkpoint)

    else:
        ingestion = commit_cloud_monthly_data(
            incoming_path
        )

        if ingestion["status"] == "ALREADY_INGESTED":
            return {
                "status": "SKIPPED",
                "reason": "Monthly data already ingested",
            }

        affected_products = ingestion["affected_products"]
        results = []

        checkpoint = {
            "batch_id": batch_id,
            "status": "IN_PROGRESS",
            "transaction_id": ingestion["transaction_id"],
            "affected_products": affected_products,
            "review_fingerprint": review_fingerprint,
            "results": results,
        }

        save_checkpoint(checkpoint)

    # Resume only genuinely successful products.
    completed = {
        item["product_code"]
        for item in results
        if (
            item.get("status") == "SUCCESS"
            and item.get("forecast_generated") is True
        )
    }

    try:
        for product_code in affected_products:

            if product_code in completed:
                print(
                    f"SKIPPED: {product_code} "
                    "(checkpoint success)",
                    flush=True,
                )
                continue

            print(
                f"REASSESSING: {product_code}",
                flush=True,
            )

            try:
                workflow = await run_workflow(
                    f"Forecast medicine {product_code} "
                    "for the next 12 months",
                    user_id="cloud_monthly_system",
                )

                evaluation = (
                    workflow.get("evaluation_result") or {}
                )

                forecast = (
                    workflow.get("forecast_result") or {}
                )

                outcome = {
                    "product_code": product_code,
                    "status": workflow.get("status"),
                    "champion_model": evaluation.get(
                        "champion_model"
                    ),
                    "challenger_model": evaluation.get(
                        "challenger_model"
                    ),
                    "registry_version": evaluation.get(
                        "registry_version"
                    ),
                    "forecast_generated": (
                        forecast.get("status") == "SUCCESS"
                    ),
                    "reason": (
                        forecast.get("reason")
                        or evaluation.get("reason")
                        or (
                            workflow.get("validation_result")
                            or {}
                        ).get("reason")
                    ),
                }

            except Exception as exc:
                outcome = {
                    "product_code": product_code,
                    "status": "FAILED",
                    "reason": str(exc),
                    "forecast_generated": False,
                }

            # Replace previous result for this product.
            results = [
                item
                for item in results
                if item["product_code"] != product_code
            ]

            results.append(outcome)

            # Persist forecasting state before checkpoint.
            save_state()

            checkpoint["results"] = results
            save_checkpoint(checkpoint)

            print(
                product_code,
                outcome["status"],
                flush=True,
            )

    finally:
        save_state()

    result_by_product = {
        item["product_code"]: item
        for item in results
    }

    successful = all(
        product_code in result_by_product
        and result_by_product[product_code].get(
            "status"
        ) == "SUCCESS"
        and result_by_product[product_code].get(
            "forecast_generated"
        ) is True
        for product_code in affected_products
    )

    checkpoint["status"] = (
        "COMPLETED"
        if successful
        else "REVIEW_REQUIRED"
    )

    checkpoint["results"] = results
    save_checkpoint(checkpoint)

    summary = {
        "status": checkpoint["status"],
        "transaction_id": checkpoint.get(
            "transaction_id"
        ),
        "affected_product_count": len(
            affected_products
        ),
        "successful_count": sum(
            1
            for product_code in affected_products
            if (
                product_code in result_by_product
                and result_by_product[product_code].get(
                    "status"
                ) == "SUCCESS"
                and result_by_product[product_code].get(
                    "forecast_generated"
                ) is True
            )
        ),
        "results": results,
    }

    record_event(
        user_id="cloud_monthly_system",
        action="CLOUD_MONTHLY_REASSESSMENT",
        status=checkpoint["status"],
        details=summary,
    )

    save_state()

    return summary
