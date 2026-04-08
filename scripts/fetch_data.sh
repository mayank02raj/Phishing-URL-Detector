#!/usr/bin/env bash
# scripts/fetch_data.sh
# Pull a fresh phishing/benign URL dataset from public sources and merge
# into a single labeled CSV at data/urls.csv.
#
# Sources:
#   - OpenPhish community feed (phishing, refreshed daily)
#   - URLhaus by abuse.ch (malicious URLs, includes phishing)
#   - Tranco top 50k sites (benign baseline)

set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[+] Fetching OpenPhish feed"
curl -fsSL https://openphish.com/feed.txt -o openphish.txt
awk 'NF {print $0",1"}' openphish.txt > phish_openphish.csv

echo "[+] Fetching URLhaus feed"
curl -fsSL https://urlhaus.abuse.ch/downloads/csv_recent/ -o urlhaus.csv
# URLhaus has comments, columns: id,dateadded,url,...
grep -v '^#' urlhaus.csv | awk -F',' 'NR>1 {gsub(/"/, "", $3); print $3",1"}' \
    > phish_urlhaus.csv

echo "[+] Fetching Tranco top 50k"
curl -fsSL https://tranco-list.eu/top-1m.csv.zip -o tranco.zip
unzip -p tranco.zip top-1m.csv | head -50000 \
    | awk -F',' '{print "http://"$2",0"}' > benign_tranco.csv

echo "[+] Merging into urls.csv"
{
    echo "url,label"
    cat phish_openphish.csv phish_urlhaus.csv benign_tranco.csv
} > urls.csv

PHISH_COUNT=$(awk -F',' 'NR>1 && $2==1' urls.csv | wc -l)
BENIGN_COUNT=$(awk -F',' 'NR>1 && $2==0' urls.csv | wc -l)
TOTAL=$((PHISH_COUNT + BENIGN_COUNT))

echo ""
echo "[+] Dataset assembled at $DATA_DIR/urls.csv"
echo "    Total:  $TOTAL"
echo "    Phish:  $PHISH_COUNT"
echo "    Benign: $BENIGN_COUNT"

# Cleanup intermediates
rm -f openphish.txt urlhaus.csv tranco.zip \
      phish_openphish.csv phish_urlhaus.csv benign_tranco.csv
