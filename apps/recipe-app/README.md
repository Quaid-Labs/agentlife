# Recipe App

A recipe organizer with meal planning, dietary tracking, and grocery list generation. Built with Express + SQLite, with a GraphQL API alongside REST endpoints.

## Features

- **Recipe CRUD** — create, read, update, delete recipes
- **Dietary filtering** — filter by tags (vegetarian, vegan, gluten-free, etc.)
- **Safe for Mom** — preset filter for diabetic-friendly + low-sodium recipes
- **Meal planning** — weekly plans with day/meal slots
- **Grocery lists** — auto-aggregated from meal plan ingredients
- **Recipe sharing** — generate shareable links
- **GraphQL API** — full schema alongside REST endpoints
- **Structured ingredients** — normalized ingredient data with amounts/units/categories

## Quick Start

```bash
npm install
npm run seed    # populate with sample recipes
npm start       # http://localhost:3000
```

## API

### REST

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/recipes` | List recipes (filters: `diet`, `safeForMom`, `maxPrepTime`) |
| GET | `/api/recipes/:id` | Get recipe with structured ingredients |
| POST | `/api/recipes` | Create recipe |
| PUT | `/api/recipes/:id` | Update recipe |
| DELETE | `/api/recipes/:id` | Delete recipe |
| GET | `/api/recipes/search?q=` | Search by title or ingredients |
| GET | `/api/dietary-labels` | List available dietary tags |
| GET | `/api/meal-plans` | List meal plans |
| POST | `/api/meal-plans` | Create meal plan |
| GET | `/api/meal-plans/:id/grocery-list` | Aggregated grocery list |
| POST | `/api/recipes/:id/share` | Generate share link |
| GET | `/api/shared/:code` | View shared recipe |

### GraphQL

```
POST /graphql
```

See `docs/api.md` for full schema and example queries.

## Development

```bash
npm run dev     # auto-reload with --watch
npm test        # run Jest test suite
make help       # see all make targets
```

## Docker

```bash
docker-compose up -d
```

## Tech Stack

- **Runtime:** Node.js 18+
- **Framework:** Express 4
- **Database:** SQLite via better-sqlite3
- **GraphQL:** Apollo Server 4
- **Tests:** Jest
