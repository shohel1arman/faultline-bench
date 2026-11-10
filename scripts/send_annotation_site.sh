#!/bin/bash
# Build the hosted annotation page with its photos and push it to a temporary branch
# (annotation-site), outside the main history, so it can be published as a link.
# Run on a machine that has data/images (the Mac or the Spark):
#   bash scripts/send_annotation_site.sh            # round 1 -> branch annotation-site
#   bash scripts/send_annotation_site.sh 2          # round 2 -> branch annotation-site-r2
set -e
cd "$(dirname "$0")/.."
ROUND="${1:-1}"
if [ "$ROUND" = "1" ]; then SITE=annotation/site; BRANCH=annotation-site; else SITE="annotation/site_r$ROUND"; BRANCH="annotation-site-r$ROUND"; fi
uv run python scripts/make_annotation_site.py --images data/images --round "$ROUND"
URL="$(git remote get-url origin)"
TMP="$(mktemp -d)"
cp -R "$SITE"/. "$TMP/"
cd "$TMP"
git init -q
git add .
git -c user.name="$(git -C "$OLDPWD" config user.name)" -c user.email="$(git -C "$OLDPWD" config user.email)" \
    commit -q -m "Annotation page with photos (temporary branch)"
git push -f -q "$URL" "HEAD:refs/heads/$BRANCH"
cd - > /dev/null
rm -rf "$TMP"
echo "pushed the annotation page to the temporary branch $BRANCH"
