/**
 * Named database queries — replaces inline SQL throughout the app.
 *
 * Three namespaces: recipes, ingredients, mealPlans. Each exposes
 * standard CRUD operations plus domain-specific helpers like
 * dietary filtering and grocery list aggregation.
 *
 * Introduced: session 10 (query consolidation pass)
 *
 * NOTE: The `db` reference is resolved lazily via getDb() so that
 * tests can swap in an in-memory database before any query runs.
 */

const { createConnection } = require('../../config/database');

/** Dietary tags that make a recipe safe for Linda (mom) */
const SAFE_FOR_MOM = ['diabetic-friendly', 'low-sodium'];

let _db = null;

/** Get or create the database connection */
function getDb() {
  if (!_db) {
    _db = createConnection();
  }
  return _db;
}

/** Allow tests to inject an in-memory database */
function setDb(db) {
  _db = db;
}

// ---------------------------------------------------------------------------
// Recipes
// ---------------------------------------------------------------------------

const recipes = {
  /**
   * Fetch all recipes, optionally filtered by dietary tags.
   *
   * @param {object} [filters]
   * @param {string[]} [filters.diet] - Required dietary tags (AND logic)
   * @param {boolean} [filters.safeForMom] - Shortcut for diabetic-friendly + low-sodium
   * @param {number} [filters.maxPrepTime] - Maximum prep time in minutes
   * @param {number} [filters.limit] - Max rows to return
   * @param {number} [filters.offset] - Rows to skip (pagination)
   */
  findAll(filters = {}) {
    const db = getDb();
    let rows = db.prepare('SELECT * FROM recipes ORDER BY created_at DESC').all();

    // Parse dietary_tags JSON for every row
    rows = rows.map((r) => ({
      ...r,
      dietary_tags: JSON.parse(r.dietary_tags || '[]'),
    }));

    // Filter: specific dietary tags (AND — every tag must be present)
    if (filters.diet && filters.diet.length > 0) {
      rows = rows.filter((r) =>
        filters.diet.every((d) => r.dietary_tags.includes(d))
      );
    }

    // Filter: safe-for-mom shortcut
    if (filters.safeForMom) {
      rows = rows.filter((r) =>
        SAFE_FOR_MOM.every((t) => r.dietary_tags.includes(t))
      );
    }

    // Filter: max prep time
    if (filters.maxPrepTime) {
      rows = rows.filter((r) => r.prep_time <= filters.maxPrepTime);
    }

    // Pagination
    const offset = filters.offset || 0;
    const limit = filters.limit || rows.length;
    rows = rows.slice(offset, offset + limit);

    return rows;
  },

  findById(id) {
    const db = getDb();
    const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(id);
    if (!row) return null;
    return { ...row, dietary_tags: JSON.parse(row.dietary_tags || '[]') };
  },

  create(data) {
    const db = getDb();
    const stmt = db.prepare(`
      INSERT INTO recipes (title, ingredients, instructions, dietary_tags, image_url, prep_time, owner_id)
      VALUES (@title, @ingredients, @instructions, @dietary_tags, @image_url, @prep_time, @owner_id)
    `);
    const result = stmt.run({
      title: data.title,
      ingredients: data.ingredients,
      instructions: data.instructions,
      dietary_tags: JSON.stringify(data.dietary_tags || []),
      image_url: data.image_url || '',
      prep_time: data.prep_time || 0,
      owner_id: data.owner_id || null,
    });
    return recipes.findById(Number(result.lastInsertRowid));
  },

  update(id, data) {
    const db = getDb();
    const existing = recipes.findById(id);
    if (!existing) return null;

    const merged = {
      title: data.title ?? existing.title,
      ingredients: data.ingredients ?? existing.ingredients,
      instructions: data.instructions ?? existing.instructions,
      dietary_tags: JSON.stringify(data.dietary_tags ?? existing.dietary_tags),
      image_url: data.image_url ?? existing.image_url,
      prep_time: data.prep_time ?? existing.prep_time,
    };

    db.prepare(`
      UPDATE recipes
      SET title = @title, ingredients = @ingredients, instructions = @instructions,
          dietary_tags = @dietary_tags, image_url = @image_url, prep_time = @prep_time
      WHERE id = @id
    `).run({ id, ...merged });

    return recipes.findById(id);
  },

  delete(id) {
    const db = getDb();
    const result = db.prepare('DELETE FROM recipes WHERE id = ?').run(id);
    return result.changes > 0;
  },

  search(query) {
    const db = getDb();
    const pattern = `%${query}%`;
    return db
      .prepare(
        'SELECT * FROM recipes WHERE title LIKE ? OR ingredients LIKE ? ORDER BY created_at DESC'
      )
      .all(pattern, pattern)
      .map((r) => ({ ...r, dietary_tags: JSON.parse(r.dietary_tags || '[]') }));
  },
};

// ---------------------------------------------------------------------------
// Ingredients
// ---------------------------------------------------------------------------

const ingredients = {
  findByRecipe(recipeId) {
    const db = getDb();
    return db
      .prepare('SELECT * FROM recipe_ingredients WHERE recipe_id = ? ORDER BY id')
      .all(recipeId);
  },

  create(data) {
    const db = getDb();
    const result = db.prepare(`
      INSERT INTO recipe_ingredients (recipe_id, name, amount, unit, category)
      VALUES (@recipe_id, @name, @amount, @unit, @category)
    `).run({
      recipe_id: data.recipe_id,
      name: data.name,
      amount: data.amount,
      unit: data.unit,
      category: data.category || 'other',
    });
    return db.prepare('SELECT * FROM recipe_ingredients WHERE id = ?')
      .get(Number(result.lastInsertRowid));
  },

  bulkCreate(recipeId, items) {
    const db = getDb();
    const stmt = db.prepare(`
      INSERT INTO recipe_ingredients (recipe_id, name, amount, unit, category)
      VALUES (@recipe_id, @name, @amount, @unit, @category)
    `);

    const inserted = [];
    db.transaction(() => {
      for (const item of items) {
        const result = stmt.run({
          recipe_id: recipeId,
          name: item.name,
          amount: item.amount,
          unit: item.unit,
          category: item.category || 'other',
        });
        inserted.push(Number(result.lastInsertRowid));
      }
    })();

    return db
      .prepare(
        `SELECT * FROM recipe_ingredients WHERE id IN (${inserted.map(() => '?').join(',')})`
      )
      .all(...inserted);
  },

  deleteByRecipe(recipeId) {
    const db = getDb();
    return db.prepare('DELETE FROM recipe_ingredients WHERE recipe_id = ?').run(recipeId);
  },
};

// ---------------------------------------------------------------------------
// Meal Plans
// ---------------------------------------------------------------------------

const mealPlans = {
  findAll() {
    const db = getDb();
    return db.prepare('SELECT * FROM meal_plans ORDER BY created_at DESC').all();
  },

  findById(id) {
    const db = getDb();
    const plan = db.prepare('SELECT * FROM meal_plans WHERE id = ?').get(id);
    if (!plan) return null;

    const items = db.prepare(`
      SELECT mpi.*, r.title AS recipe_title, r.dietary_tags, r.prep_time
      FROM meal_plan_items mpi
      JOIN recipes r ON r.id = mpi.recipe_id
      WHERE mpi.plan_id = ?
      ORDER BY
        CASE mpi.day_of_week
          WHEN 'monday' THEN 1 WHEN 'tuesday' THEN 2 WHEN 'wednesday' THEN 3
          WHEN 'thursday' THEN 4 WHEN 'friday' THEN 5 WHEN 'saturday' THEN 6
          WHEN 'sunday' THEN 7
        END,
        CASE mpi.meal_type
          WHEN 'breakfast' THEN 1 WHEN 'lunch' THEN 2 WHEN 'dinner' THEN 3
          WHEN 'snack' THEN 4
        END
    `).all(id);

    return { ...plan, items };
  },

  create(data) {
    const db = getDb();
    const result = db.prepare(
      'INSERT INTO meal_plans (week_start, name) VALUES (@week_start, @name)'
    ).run({ week_start: data.week_start, name: data.name });
    return mealPlans.findById(Number(result.lastInsertRowid));
  },

  addItem(planId, data) {
    const db = getDb();
    const result = db.prepare(`
      INSERT INTO meal_plan_items (plan_id, recipe_id, day_of_week, meal_type)
      VALUES (@plan_id, @recipe_id, @day_of_week, @meal_type)
    `).run({
      plan_id: planId,
      recipe_id: data.recipe_id,
      day_of_week: data.day_of_week,
      meal_type: data.meal_type,
    });
    return db.prepare('SELECT * FROM meal_plan_items WHERE id = ?')
      .get(Number(result.lastInsertRowid));
  },

  removeItem(itemId) {
    const db = getDb();
    return db.prepare('DELETE FROM meal_plan_items WHERE id = ?').run(itemId).changes > 0;
  },

  /**
   * Aggregate a grocery list from all recipes in a meal plan.
   * Groups by ingredient name, sums amounts, and includes categories.
   */
  getGroceryList(planId) {
    const db = getDb();
    return db.prepare(`
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
    `).all(planId);
  },
};

module.exports = { recipes, ingredients, mealPlans, SAFE_FOR_MOM, getDb, setDb };
