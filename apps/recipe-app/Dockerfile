# =============================================================================
# Dockerfile for Recipe App
#
# Uses Node 18 Alpine for a minimal image (~120MB). better-sqlite3
# needs build tools for the native addon, but npm ci handles that
# via prebuild-install (downloads prebuilt binary when available).
#
# Introduced: session 12
# =============================================================================

FROM node:18-alpine

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY package*.json ./

# Install production dependencies only
RUN npm ci --production

# Copy application source
COPY . .

# Create data directory for SQLite persistence (mounted as volume)
RUN mkdir -p /app/data

EXPOSE 3000

CMD ["node", "server.js"]
