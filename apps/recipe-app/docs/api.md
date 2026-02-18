# Recipe App API Documentation

## Authentication

All mutation endpoints (POST, PUT, DELETE) require a valid JWT token. Read
endpoints (GET) are public.

### Register

```
POST /api/auth/register
Content-Type: application/json

{
  "username": "maya",
  "email": "maya@example.com",
  "password": "securepassword123",
  "display_name": "Maya Chen",
  "dietary_preferences": ["vegetarian"]
}
```

**Response** `201 Created`
```json
{
  "user": {
    "id": 1,
    "username": "maya",
    "email": "maya@example.com",
    "display_name": "Maya Chen",
    "dietary_preferences": ["vegetarian"]
  },
  "token": "eyJhbGciOi..."
}
```

### Login

```
POST /api/auth/login
Content-Type: application/json

{ "username": "maya", "password": "securepassword123" }
```

**Response** `200 OK`
```json
{ "token": "eyJhbGciOi...", "user": { "id": 1, "username": "maya" } }
```

### Using the Token

Include the JWT in the `Authorization` header:

```
Authorization: Bearer eyJhbGciOi...
```

Tokens expire after 24 hours.

---

## REST Endpoints

### Recipes

#### List Recipes

```
GET /api/recipes
GET /api/recipes?diet=vegetarian
GET /api/recipes?diet=diabetic-friendly,low-sodium
GET /api/recipes?safeForMom=true
GET /api/recipes?maxPrepTime=30
GET /api/recipes?limit=10&offset=0
```

**Response** `200 OK`
```json
[
  {
    "id": 1,
    "title": "Chickpea Curry",
    "ingredients": "1 can chickpeas, 1 can coconut milk, ...",
    "instructions": "Sauté onion, garlic, and ginger...",
    "dietary_tags": ["vegetarian", "vegan", "gluten-free", "dairy-free"],
    "image_url": "",
    "prep_time": 35,
    "created_at": "2026-02-16T12:00:00.000Z"
  }
]
```

#### Get Recipe by ID

```
GET /api/recipes/:id
```

**Response** `200 OK` or `404 Not Found`

#### Create Recipe (auth required)

```
POST /api/recipes
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "New Recipe",
  "ingredients": "flour, sugar, butter",
  "instructions": "Mix and bake.",
  "dietary_tags": ["vegetarian"],
  "image_url": "",
  "prep_time": 30
}
```

**Response** `201 Created`

#### Update Recipe (auth required)

```
PUT /api/recipes/:id
Authorization: Bearer <token>
Content-Type: application/json

{ "title": "Updated Title", "prep_time": 25 }
```

Partial updates are supported -- only include fields you want to change.

**Response** `200 OK` or `404 Not Found`

#### Delete Recipe (auth required)

```
DELETE /api/recipes/:id
Authorization: Bearer <token>
```

**Response** `204 No Content` or `404 Not Found`

#### Search Recipes

```
GET /api/recipes/search?q=curry
```

Searches recipe titles and ingredient lists. Case-insensitive.

**Response** `200 OK` with array of matching recipes.

### Meal Plans

#### List Meal Plans

```
GET /api/meal-plans
```

#### Get Meal Plan (with items)

```
GET /api/meal-plans/:id
```

**Response** includes the plan metadata plus all items with recipe details:

```json
{
  "id": 1,
  "week_start": "2026-02-16",
  "name": "Family Meal Plan",
  "items": [
    {
      "id": 1,
      "day_of_week": "monday",
      "meal_type": "dinner",
      "recipe_id": 6,
      "recipe_title": "Grilled Salmon with Herbs",
      "prep_time": 30
    }
  ]
}
```

#### Create Meal Plan (auth required)

```
POST /api/meal-plans
Authorization: Bearer <token>
Content-Type: application/json

{ "week_start": "2026-02-16", "name": "This Week" }
```

#### Add Item to Meal Plan (auth required)

```
POST /api/meal-plans/:id/items
Authorization: Bearer <token>
Content-Type: application/json

{ "recipe_id": 3, "day_of_week": "wednesday", "meal_type": "dinner" }
```

Valid days: monday through sunday. Valid meal types: breakfast, lunch, dinner, snack.

#### Remove Item from Meal Plan (auth required)

```
DELETE /api/meal-plans/:id/items/:itemId
Authorization: Bearer <token>
```

#### Get Grocery List for a Meal Plan

```
GET /api/meal-plans/:id/grocery-list
```

Aggregates all ingredients across the plan's recipes, grouped by category:

```json
[
  {
    "name": "olive oil",
    "unit": "tbsp",
    "category": "condiments",
    "total_amount": 6,
    "recipe_count": 3,
    "from_recipes": "Grilled Salmon with Herbs,Lentil Soup,Roasted Vegetable Medley"
  }
]
```

### Recipe Sharing

#### Create Share Link

```
POST /api/recipes/:id/share
Authorization: Bearer <token>
```

**Response** `201 Created`
```json
{ "code": "abc123xy", "url": "/shared/abc123xy" }
```

#### View Shared Recipe (no auth)

```
GET /api/shared/:code
```

---

## GraphQL

**Endpoint:** `POST /graphql`

The GraphQL API mirrors the REST endpoints. Authentication works the same
way -- pass the JWT in the `Authorization` header.

### Example Queries

```graphql
# List all vegetarian recipes
query {
  recipes(diet: ["vegetarian"]) {
    id
    title
    dietary_tags
    prep_time
  }
}

# Get a single recipe with its structured ingredients
query {
  recipe(id: 5) {
    title
    instructions
    ingredients {
      name
      amount
      unit
      category
    }
  }
}

# Search recipes
query {
  searchRecipes(query: "curry") {
    id
    title
    dietary_tags
  }
}

# Get a meal plan with grocery list
query {
  mealPlan(id: 1) {
    name
    week_start
    items {
      day_of_week
      meal_type
      recipe {
        title
        prep_time
      }
    }
    groceryList {
      name
      total_amount
      unit
      category
    }
  }
}
```

### Example Mutations

```graphql
# Create a recipe
mutation {
  createRecipe(input: {
    title: "Quick Salad"
    ingredients: "mixed greens, cherry tomatoes, cucumber, olive oil, lemon juice"
    instructions: "Toss everything together. Drizzle with olive oil and lemon."
    dietary_tags: ["vegetarian", "vegan", "gluten-free"]
    prep_time: 10
  }) {
    id
    title
  }
}

# Create a meal plan and add items
mutation {
  createMealPlan(input: { week_start: "2026-02-23", name: "Next Week" }) {
    id
  }
}

mutation {
  addMealPlanItem(planId: 2, input: {
    recipe_id: 1
    day_of_week: "monday"
    meal_type: "dinner"
  }) {
    id
    day_of_week
  }
}
```

### Schema Overview

```graphql
type Recipe {
  id: Int!
  title: String!
  ingredients: String!       # Raw text
  instructions: String!
  dietary_tags: [String!]!
  image_url: String
  prep_time: Int
  created_at: String
  structuredIngredients: [Ingredient!]!   # Parsed entries
}

type Ingredient {
  id: Int!
  name: String!
  amount: Float!
  unit: String!
  category: String
}

type MealPlan {
  id: Int!
  name: String!
  week_start: String!
  items: [MealPlanItem!]!
  groceryList: [GroceryItem!]!
}

type MealPlanItem {
  id: Int!
  day_of_week: String!
  meal_type: String!
  recipe: Recipe!
}

type GroceryItem {
  name: String!
  total_amount: Float!
  unit: String!
  category: String
  recipe_count: Int!
  from_recipes: String
}
```

---

## Rate Limiting

All endpoints are rate-limited to **100 requests per 15-minute window** per IP.
Exceeding the limit returns `429 Too Many Requests`.

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "status": 404,
    "message": "Recipe not found"
  }
}
```

Common status codes: 400 (validation), 401 (unauthorized), 404 (not found),
429 (rate limited), 500 (server error).
