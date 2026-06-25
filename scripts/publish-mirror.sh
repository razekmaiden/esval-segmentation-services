#!/usr/bin/env bash
# Publica el árbol actual (rama runpod-implementation) al repositorio espejo en Innervycs.
# Crea un único commit en main sin historial previo.
#
# Uso (desde la raíz del repo, en runpod-implementation):
#   ./scripts/publish-mirror.sh
#   MIRROR_GITHUB_TOKEN=<pat> ./scripts/publish-mirror.sh
#
# Variables:
#   MIRROR_ORG          default: Innervycs
#   MIRROR_GITHUB_TOKEN token con permiso repo en Innervycs (o gh auth token)

set -euo pipefail

MIRROR_ORG="${MIRROR_ORG:-Innervycs}"
ORIGIN_OWNER="${ORIGIN_OWNER:-razekmaiden}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="$(basename "$ROOT")"
MIRROR_NAME="${REPO_NAME}_mirror"
SOURCE_SHA="$(git -C "$ROOT" rev-parse HEAD)"
SOURCE_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"

TOKEN="${MIRROR_GITHUB_TOKEN:-${GITHUB_TOKEN:-}}"
if [[ -z "$TOKEN" ]] && command -v gh &>/dev/null; then
    TOKEN="$(gh auth token 2>/dev/null || true)"
fi
[[ -n "$TOKEN" ]] || { echo "❌ Falta MIRROR_GITHUB_TOKEN o gh auth login"; exit 1; }

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

log() { echo "[mirror] $*"; }

log "Origen:  ${ORIGIN_OWNER}/${REPO_NAME}@${SOURCE_SHA} (${SOURCE_BRANCH})"
log "Destino: ${MIRROR_ORG}/${MIRROR_NAME} → main"

rsync -a \
    --exclude '.git' \
    --exclude 'node_modules' \
    --exclude '.next' \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude '.pytest_cache' \
    --exclude 'scripts/publish-mirror.sh' \
    --exclude 'scripts/mirror' \
    --exclude '.github/workflows/sync-mirror.yml' \
    "$ROOT/" "$WORKDIR/"

rewrite_urls() {
    local f="$1"
    [[ -f "$f" ]] || return 0
    sed -i \
        -e 's|git@github.com:razekmaiden/esval-web\.git|git@github.com:Innervycs/esval-web_mirror.git|g' \
        -e 's|git@github.com:razekmaiden/esval-segmentation-services\.git|git@github.com:Innervycs/esval-segmentation-services_mirror.git|g' \
        -e 's|git@github.com:razekmaiden/esval-sii-api\.git|git@github.com:Innervycs/esval-sii-api_mirror.git|g' \
        -e 's|https://github.com/razekmaiden/esval-web|https://github.com/Innervycs/esval-web_mirror|g' \
        -e 's|https://github.com/razekmaiden/esval-segmentation-services|https://github.com/Innervycs/esval-segmentation-services_mirror|g' \
        -e 's|https://github.com/razekmaiden/esval-sii-api|https://github.com/Innervycs/esval-sii-api_mirror|g' \
        -e 's|razekmaiden/esval-web|Innervycs/esval-web_mirror|g' \
        -e 's|razekmaiden/esval-segmentation-services|Innervycs/esval-segmentation-services_mirror|g' \
        -e 's|razekmaiden/esval-sii-api|Innervycs/esval-sii-api_mirror|g' \
        -e 's|rama `runpod-implementation`|rama `main`|g' \
        -e 's|rama runpod-implementation|rama main|g' \
        -e 's|branch runpod-implementation|branch main|g' \
        -e 's|git checkout runpod-implementation|git checkout main|g' \
        -e 's|git pull origin runpod-implementation|git pull origin main|g' \
        -e 's|-b runpod-implementation |-b main |g' \
        -e 's|--branch runpod-implementation|--branch main|g' \
        -e 's|runpod-implementation|main|g' \
        "$f"
}

while IFS= read -r -d '' file; do
    case "$file" in
        *.md|*.sh|*.yml|*.yaml|*.ts|*.tsx|*.js|*.json|*.example|.env.example)
            rewrite_urls "$file"
            ;;
    esac
done < <(find "$WORKDIR" -type f -print0)

cp "$ROOT/scripts/mirror/LICENSE" "$WORKDIR/LICENSE"
cp "$ROOT/scripts/mirror/MIRROR.md" "$WORKDIR/MIRROR.md"

cd "$WORKDIR"
git init -q
git config user.name "Innervycs Mirror Bot"
git config user.email "info@innervycs.com"
git checkout -q -b main
git add -A
git commit -q -m "Mirror snapshot from ${ORIGIN_OWNER}/${REPO_NAME}@${SOURCE_SHA}"

MIRROR_URL="https://x-access-token:${TOKEN}@github.com/${MIRROR_ORG}/${MIRROR_NAME}.git"
API="https://api.github.com/repos/${MIRROR_ORG}/${MIRROR_NAME}"

if ! curl -sf -H "Authorization: Bearer ${TOKEN}" -H "Accept: application/vnd.github+json" "$API" >/dev/null; then
    log "Creando repositorio privado ${MIRROR_ORG}/${MIRROR_NAME}..."
    curl -sf -X POST \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/orgs/${MIRROR_ORG}/repos" \
        -d "{\"name\":\"${MIRROR_NAME}\",\"private\":true,\"description\":\"Mirror de despliegue ESVAL (${REPO_NAME})\"}" >/dev/null
fi

git remote add origin "$MIRROR_URL"
git push --force origin main

log "✅ Publicado ${MIRROR_ORG}/${MIRROR_NAME}@main (${SOURCE_SHA})"
