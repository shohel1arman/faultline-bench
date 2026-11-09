#!/bin/bash
# Build the hosted annotation page with its photos and push it to a temporary branch
# (annotation-site), outside the main history, so it can be published as a link.
# Run on a machine that has data/images (the Mac or the Spark):
#   bash scripts/send_annotation_site.sh
set -e
cd "$(dirname "$0")/.."
uv run python scripts/make_annotation_site.py --images data/images
URL="$(git remote get-url origin)"
TMP="$(mktemp -d)"
cp -R annotation/site/. "$TMP/"
cd "$TMP"
git init -q
git add .
git -c user.name="$(git -C "$OLDPWD" config user.name)" -c user.email="$(git -C "$OLDPWD" config user.email)" \
    commit -q -m "Annotation page with photos (temporary branch)"
git push -f -q "$URL" HEAD:refs/heads/annotation-site
cd - > /dev/null
rm -rf "$TMP"
echo "pushed the annotation page to the temporary branch annotation-site"
