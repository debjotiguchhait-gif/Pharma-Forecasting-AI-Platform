HITL FINAL INTEGRATION PATCH (v3)

1. Extract ZIP into the EXISTING project root, preserving tools/ directory.
   Do not replace your entire forecasting project with this patch.
2. Build: gcloud.cmd builds submit . --config=cloudbuild-hitl.yaml --project=pharma-auto-forecasting
3. Deploy: gcloud.cmd run deploy pharma-hitl-dashboard --image=asia-south1-docker.pkg.dev/pharma-auto-forecasting/pharma-forecasting/hitl-dashboard:v3 --region=asia-south1 --project=pharma-auto-forecasting
4. Open existing IAP-protected dashboard. Check identity, queue, and submit ONE noncritical real observation only after reviewing it. Confirm count decreases and GCS record exists.
5. Forecasting job code is updated locally but the Cloud Run JOB image must be rebuilt/redeployed with your ORIGINAL forecasting Dockerfile/build configuration; dashboard deployment does NOT update that job. Do not run the job until its updated image is deployed.

Safety: No IAP settings, reviewer IAM, or production GCS data are changed by this ZIP.
Historical audit queue is a snapshot: decisions disappear from the queue, but statuses only update after a forecasting rerun.
The current project ZIP contained only nine integration files, NOT the full forecasting code or build setup. Consequently a complete end-to-end forecasting run has NOT been validated here.
