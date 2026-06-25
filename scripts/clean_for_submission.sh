#!/usr/bin/env bash
# Strips sensitive / personal files before zipping the project for
# submission or sharing. Run this from the project root.
#
# Usage:
#   bash scripts/clean_for_submission.sh           # default (safe)
#   bash scripts/clean_for_submission.sh --strict  # also removes credentials.json + caches
#
# What it always removes:
#   - token.json          (your cached Gmail OAuth session — THE main risk)
#   - .env, .env.local    (API keys)
#   - generated CSVs      (campaign_emails.csv, lead_scoring_results.csv,
#                          qualified_opportunities.csv)
#   - macOS .DS_Store noise
#
# Extra with --strict:
#   - credentials.json    (your Google Cloud OAuth client ID)
#   - data/               (your local BeamData PDF knowledge base)
#   - __pycache__/        directories

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "Cleaning project at: $ROOT"

remove() {
    if [ -e "$1" ]; then
        rm -rf "$1"
        echo "  removed  $1"
    fi
}

# Always
remove token.json
remove .env
remove .env.local
remove campaign_emails.csv
remove lead_scoring_results.csv
remove qualified_opportunities.csv
remove proposal_ready_opportunities.csv
find . -name ".DS_Store" -type f -delete 2>/dev/null || true

if [ "${1:-}" = "--strict" ]; then
    echo "--- strict mode ---"
    remove credentials.json
    remove data
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo "  removed  all __pycache__/ directories"
fi

echo
echo "Done. Remaining sensitive surface to consider:"
echo "  - credentials.json  (kept by default; remove with --strict if you don't"
echo "                       want your Google Cloud OAuth client ID in the zip)"
echo "  - revoke the app's grant at https://myaccount.google.com/permissions"
echo "    if you want to be extra safe after submission."
