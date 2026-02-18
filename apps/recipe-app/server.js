const express = require('express');
const { ApolloServer } = require('@apollo/server');
const { expressMiddleware } = require('@apollo/server/express4');
const db = require('./database');
const { DIETARY_LABELS, SAFE_FOR_MOM } = require('./database');
const { typeDefs } = require('./schema');
const { resolvers } = require('./resolvers');
const crypto = require('crypto');
const jwt = require('jsonwebtoken');
const { errorHandler, notFoundHandler } = require('./src/middleware/errorHandler');
const { requestLogger } = require('./src/middleware/logging');
const { rateLimiter } = require('./src/middleware/rateLimiter');
const { requireAuth } = require('./src/middleware/auth');
const authConfig = require('./config/auth');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static('public'));
app.use(requestLogger);

// Rate limit API routes — 100 requests per 15-minute window per IP
app.use('/api', rateLimiter({ windowMs: 15 * 60 * 1000, max: 100 }));

// ---- Health check (used by Docker healthcheck) ----

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// ---- Recipes ----

// Get all recipes (with optional dietary filter)
app.get('/api/recipes', (req, res) => {
  let recipes = db.prepare('SELECT * FROM recipes ORDER BY created_at DESC').all();

  // Filter by dietary tags
  if (req.query.safeForMom === 'true') {
    recipes = recipes.filter(r => {
      const tags = JSON.parse(r.dietary_tags || '[]');
      return SAFE_FOR_MOM.every(t => tags.includes(t));
    });
  } else if (req.query.diet) {
    const required = req.query.diet.split(',');
    recipes = recipes.filter(r => {
      const tags = JSON.parse(r.dietary_tags || '[]');
      return required.every(t => tags.includes(t));
    });
  }

  // Filter by max prep time
  if (req.query.maxPrepTime) {
    const maxTime = parseInt(req.query.maxPrepTime, 10);
    recipes = recipes.filter(r => r.prep_time <= maxTime);
  }

  res.json(recipes);
});

// Search recipes
app.get('/api/recipes/search', (req, res) => {
  const { q } = req.query;
  if (!q) return res.json([]);
  const recipes = db.prepare(
    'SELECT * FROM recipes WHERE title LIKE ? OR ingredients LIKE ?'
  ).all(`%${q}%`, `%${q}%`);
  res.json(recipes);
});

// Get single recipe (with structured ingredients)
app.get('/api/recipes/:id', (req, res) => {
  const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(req.params.id);
  if (!recipe) return res.status(404).json({ error: 'Recipe not found' });
  const ingredients = db.prepare(
    'SELECT * FROM recipe_ingredients WHERE recipe_id = ? ORDER BY id'
  ).all(req.params.id);
  res.json({ ...recipe, structuredIngredients: ingredients });
});

// Create recipe
app.post('/api/recipes', (req, res) => {
  const { title, ingredients, instructions, dietary_tags, image_url, prep_time } = req.body;
  if (!title || !ingredients || !instructions) {
    return res.status(400).json({ error: 'Title, ingredients, and instructions required' });
  }
  const tags = Array.isArray(dietary_tags) ? JSON.stringify(dietary_tags) : '[]';
  const result = db.prepare(
    'INSERT INTO recipes (title, ingredients, instructions, dietary_tags, image_url, prep_time) VALUES (?, ?, ?, ?, ?, ?)'
  ).run(title, ingredients, instructions, tags, image_url || '', prep_time || 0);
  const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(result.lastInsertRowid);
  res.status(201).json(recipe);
});

// Update recipe
app.put('/api/recipes/:id', (req, res) => {
  const { title, ingredients, instructions, dietary_tags, image_url, prep_time } = req.body;
  const existing = db.prepare('SELECT * FROM recipes WHERE id = ?').get(req.params.id);
  if (!existing) return res.status(404).json({ error: 'Recipe not found' });
  const tags = dietary_tags ? JSON.stringify(dietary_tags) : existing.dietary_tags;
  db.prepare(
    'UPDATE recipes SET title = ?, ingredients = ?, instructions = ?, dietary_tags = ?, image_url = ?, prep_time = ? WHERE id = ?'
  ).run(
    title || existing.title,
    ingredients || existing.ingredients,
    instructions || existing.instructions,
    tags,
    image_url !== undefined ? image_url : existing.image_url,
    prep_time !== undefined ? prep_time : existing.prep_time,
    req.params.id
  );
  const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(req.params.id);
  res.json(recipe);
});

// Delete recipe
app.delete('/api/recipes/:id', (req, res) => {
  const existing = db.prepare('SELECT * FROM recipes WHERE id = ?').get(req.params.id);
  if (!existing) return res.status(404).json({ error: 'Recipe not found' });
  db.prepare('DELETE FROM recipes WHERE id = ?').run(req.params.id);
  res.json({ message: 'Recipe deleted' });
});

// Add structured ingredients to a recipe
app.post('/api/recipes/:id/ingredients', (req, res) => {
  const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(req.params.id);
  if (!recipe) return res.status(404).json({ error: 'Recipe not found' });
  const { ingredients } = req.body;
  if (!Array.isArray(ingredients) || ingredients.length === 0) {
    return res.status(400).json({ error: 'ingredients must be a non-empty array' });
  }
  const stmt = db.prepare(
    'INSERT INTO recipe_ingredients (recipe_id, name, amount, unit, category) VALUES (?, ?, ?, ?, ?)'
  );
  for (const ing of ingredients) {
    stmt.run(req.params.id, ing.name, ing.amount, ing.unit, ing.category || 'other');
  }
  const result = db.prepare('SELECT * FROM recipe_ingredients WHERE recipe_id = ?').all(req.params.id);
  res.status(201).json(result);
});

// Dietary labels endpoint
app.get('/api/dietary-labels', (req, res) => {
  res.json(DIETARY_LABELS);
});

// ---- Recipe Sharing ----

// Create a share link for a recipe
app.post('/api/recipes/:id/share', (req, res) => {
  const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(req.params.id);
  if (!recipe) return res.status(404).json({ error: 'Recipe not found' });

  // Check if already shared
  const existing = db.prepare('SELECT * FROM recipe_shares WHERE recipe_id = ?').get(req.params.id);
  if (existing) return res.json({ code: existing.code, url: `/shared/${existing.code}` });

  const code = Math.random().toString(36).substring(2, 10);
  db.prepare('INSERT INTO recipe_shares (recipe_id, code) VALUES (?, ?)').run(req.params.id, code);
  res.status(201).json({ code, url: `/shared/${code}` });
});

// View a shared recipe by code
app.get('/api/shared/:code', (req, res) => {
  const share = db.prepare('SELECT * FROM recipe_shares WHERE code = ?').get(req.params.code);
  if (!share) return res.status(404).json({ error: 'Share link not found' });
  const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(share.recipe_id);
  if (!recipe) return res.status(404).json({ error: 'Recipe not found' });
  res.json(recipe);
});

// ---- Meal Plans ----

// List meal plans
app.get('/api/meal-plans', (req, res) => {
  const plans = db.prepare('SELECT * FROM meal_plans ORDER BY created_at DESC').all();
  res.json(plans);
});

// Get single meal plan with items
app.get('/api/meal-plans/:id', (req, res) => {
  const plan = db.prepare('SELECT * FROM meal_plans WHERE id = ?').get(req.params.id);
  if (!plan) return res.status(404).json({ error: 'Meal plan not found' });
  const items = db.prepare(`
    SELECT mpi.*, r.title AS recipe_title, r.prep_time
    FROM meal_plan_items mpi
    JOIN recipes r ON r.id = mpi.recipe_id
    WHERE mpi.plan_id = ?
    ORDER BY mpi.day_of_week, mpi.meal_type
  `).all(req.params.id);
  res.json({ ...plan, items });
});

// Create meal plan
app.post('/api/meal-plans', (req, res) => {
  const { week_start, name } = req.body;
  if (!week_start || !name) {
    return res.status(400).json({ error: 'week_start and name required' });
  }
  const result = db.prepare('INSERT INTO meal_plans (week_start, name) VALUES (?, ?)').run(week_start, name);
  const plan = db.prepare('SELECT * FROM meal_plans WHERE id = ?').get(result.lastInsertRowid);
  res.status(201).json(plan);
});

// Add item to meal plan
app.post('/api/meal-plans/:id/items', (req, res) => {
  const plan = db.prepare('SELECT * FROM meal_plans WHERE id = ?').get(req.params.id);
  if (!plan) return res.status(404).json({ error: 'Meal plan not found' });
  const { recipe_id, day_of_week, meal_type } = req.body;
  if (!recipe_id || !day_of_week || !meal_type) {
    return res.status(400).json({ error: 'recipe_id, day_of_week, and meal_type required' });
  }
  const result = db.prepare(
    'INSERT INTO meal_plan_items (plan_id, recipe_id, day_of_week, meal_type) VALUES (?, ?, ?, ?)'
  ).run(req.params.id, recipe_id, day_of_week, meal_type);
  const item = db.prepare('SELECT * FROM meal_plan_items WHERE id = ?').get(result.lastInsertRowid);
  res.status(201).json(item);
});

// Remove item from meal plan
app.delete('/api/meal-plans/:planId/items/:itemId', (req, res) => {
  const result = db.prepare('DELETE FROM meal_plan_items WHERE id = ? AND plan_id = ?')
    .run(req.params.itemId, req.params.planId);
  if (result.changes === 0) return res.status(404).json({ error: 'Item not found' });
  res.json({ message: 'Item removed' });
});

// Get grocery list for a meal plan (aggregated ingredients)
app.get('/api/meal-plans/:id/grocery-list', (req, res) => {
  const plan = db.prepare('SELECT * FROM meal_plans WHERE id = ?').get(req.params.id);
  if (!plan) return res.status(404).json({ error: 'Meal plan not found' });
  const groceryList = db.prepare(`
    SELECT
      ri.name,
      ri.unit,
      ri.category,
      SUM(ri.amount) AS total_amount,
      COUNT(DISTINCT mpi.recipe_id) AS recipe_count,
      GROUP_CONCAT(DISTINCT r.title) AS from_recipes
    FROM meal_plan_items mpi
    JOIN recipe_ingredients ri ON ri.recipe_id = mpi.recipe_id
    JOIN recipes r ON r.id = mpi.recipe_id
    WHERE mpi.plan_id = ?
    GROUP BY ri.name, ri.unit, ri.category
    ORDER BY ri.category, ri.name
  `).all(req.params.id);
  res.json(groceryList);
});

// Nutrition stub — returns placeholder until Edamam API configured
app.get('/api/recipes/:id/nutrition', (req, res) => {
  const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(req.params.id);
  if (!recipe) return res.status(404).json({ error: 'Recipe not found' });
  res.json({
    recipe_id: recipe.id,
    source: 'stub',
    message: 'Nutrition API not configured. Set EDAMAM_APP_ID and EDAMAM_APP_KEY in .env',
    estimated: { calories: null, protein: null, carbs: null, fat: null },
  });
});

// ---- Authentication (session 18) ----

// Hash a password with PBKDF2
function hashPassword(password) {
  const salt = crypto.randomBytes(authConfig.password.saltLength).toString('hex');
  const hash = crypto.pbkdf2Sync(
    password, salt,
    authConfig.password.iterations,
    authConfig.password.keyLength,
    authConfig.password.digest
  ).toString('hex');
  return `${hash}:${salt}`;
}

// Verify a password against a stored hash
function verifyPassword(password, stored) {
  const [hash, salt] = stored.split(':');
  const test = crypto.pbkdf2Sync(
    password, salt,
    authConfig.password.iterations,
    authConfig.password.keyLength,
    authConfig.password.digest
  ).toString('hex');
  return hash === test;
}

// Register a new user
app.post('/api/auth/register', (req, res) => {
  const { username, email, password, display_name, dietary_preferences } = req.body;
  if (!username || !email || !password) {
    return res.status(400).json({ error: 'username, email, and password required' });
  }
  if (password.length < 8) {
    return res.status(400).json({ error: 'Password must be at least 8 characters' });
  }

  // Check for existing user
  const existing = db.prepare('SELECT id FROM users WHERE username = ? OR email = ?').get(username, email);
  if (existing) {
    return res.status(409).json({ error: 'Username or email already taken' });
  }

  const password_hash = hashPassword(password);
  const prefs = Array.isArray(dietary_preferences) ? JSON.stringify(dietary_preferences) : '[]';
  const result = db.prepare(
    'INSERT INTO users (username, email, password_hash, display_name, dietary_preferences) VALUES (?, ?, ?, ?, ?)'
  ).run(username, email, password_hash, display_name || '', prefs);

  const user = db.prepare('SELECT id, username, email, display_name, dietary_preferences, created_at FROM users WHERE id = ?')
    .get(result.lastInsertRowid);

  const token = jwt.sign(
    { userId: user.id, username: user.username },
    authConfig.jwt.secret,
    { expiresIn: authConfig.jwt.expiresIn, algorithm: authConfig.jwt.algorithm }
  );

  res.status(201).json({ user, token });
});

// Login
app.post('/api/auth/login', (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({ error: 'username and password required' });
  }

  const user = db.prepare('SELECT * FROM users WHERE username = ?').get(username);
  if (!user || !verifyPassword(password, user.password_hash)) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  const token = jwt.sign(
    { userId: user.id, username: user.username },
    authConfig.jwt.secret,
    { expiresIn: authConfig.jwt.expiresIn, algorithm: authConfig.jwt.algorithm }
  );

  res.json({
    token,
    user: {
      id: user.id,
      username: user.username,
      email: user.email,
      display_name: user.display_name,
    },
  });
});

// Get current user profile (requires auth)
app.get('/api/auth/me', requireAuth, (req, res) => {
  const user = db.prepare(
    'SELECT id, username, email, display_name, dietary_preferences, created_at FROM users WHERE id = ?'
  ).get(req.user.userId);
  if (!user) return res.status(404).json({ error: 'User not found' });
  res.json(user);
});

// ---- GraphQL ----

async function startServer() {
  const apollo = new ApolloServer({ typeDefs, resolvers });
  await apollo.start();
  app.use('/graphql', expressMiddleware(apollo));

  // Error handling (must be after GraphQL middleware)
  app.use(notFoundHandler);
  app.use(errorHandler);

  app.listen(PORT, () => {
    console.log(`Recipe app running at http://localhost:${PORT}`);
    console.log(`GraphQL endpoint: http://localhost:${PORT}/graphql`);
  });
}

startServer();
