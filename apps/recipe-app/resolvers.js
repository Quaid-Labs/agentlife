/**
 * GraphQL Resolvers
 *
 * Introduced: session 12 (GraphQL pivot)
 *
 * TODO(session-16): N+1 query bug in Recipe.ingredientList.
 * Fires a separate SQL query per recipe in a list. 100 recipes = 101 queries.
 * Fix: DataLoader that batches into WHERE recipe_id IN (...).
 * Discovered during session 16 bug bash — not yet fixed.
 *
 * TODO(session-16): Missing authorization on updateRecipe and deleteRecipe.
 * Any user can modify/delete any recipe. Needs auth middleware.
 */

const db = require('./database');
const { DIETARY_LABELS, SAFE_FOR_MOM } = require('./database');

const resolvers = {
  Query: {
    recipes: (_, { dietFilter, safeForMom, maxPrepTime } = {}) => {
      let rows = db.prepare('SELECT * FROM recipes ORDER BY created_at DESC').all();

      if (safeForMom) {
        rows = rows.filter(r => {
          const tags = JSON.parse(r.dietary_tags || '[]');
          return SAFE_FOR_MOM.every(t => tags.includes(t));
        });
      } else if (dietFilter && dietFilter.length > 0) {
        rows = rows.filter(r => {
          const tags = JSON.parse(r.dietary_tags || '[]');
          return dietFilter.every(t => tags.includes(t));
        });
      }

      if (maxPrepTime) {
        rows = rows.filter(r => r.prep_time <= maxPrepTime);
      }

      return rows;
    },

    recipe: (_, { id }) => {
      return db.prepare('SELECT * FROM recipes WHERE id = ?').get(id) || null;
    },

    searchRecipes: (_, { query }) => {
      return db.prepare(
        'SELECT * FROM recipes WHERE title LIKE ? OR ingredients LIKE ?'
      ).all(`%${query}%`, `%${query}%`);
    },

    recipeByShareCode: (_, { code }) => {
      const share = db.prepare('SELECT * FROM recipe_shares WHERE code = ?').get(code);
      if (!share) return null;
      return db.prepare('SELECT * FROM recipes WHERE id = ?').get(share.recipe_id) || null;
    },

    dietaryLabels: () => DIETARY_LABELS,

    mealPlans: () => {
      return db.prepare('SELECT * FROM meal_plans ORDER BY week_start DESC').all();
    },

    mealPlan: (_, { id }) => {
      return db.prepare('SELECT * FROM meal_plans WHERE id = ?').get(id) || null;
    },
  },

  Mutation: {
    createRecipe: (_, args) => {
      const result = db.prepare(`
        INSERT INTO recipes (title, ingredients, instructions, dietary_tags, image_url, prep_time, owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `).run(
        args.title,
        args.ingredients,
        args.instructions,
        JSON.stringify(args.dietary_tags || []),
        args.image_url || '',
        args.prep_time || 0,
        args.owner_id || null
      );
      return db.prepare('SELECT * FROM recipes WHERE id = ?').get(result.lastInsertRowid);
    },

    updateRecipe: (_, args) => {
      // TODO(session-16): No authorization check — any user can update any recipe.
      // Should verify the requesting user's ID matches the recipe's owner_id.
      // Filed during session 16 bug bash — will fix when auth is added.
      const existing = db.prepare('SELECT * FROM recipes WHERE id = ?').get(args.id);
      if (!existing) return null;

      db.prepare(`
        UPDATE recipes SET title = ?, ingredients = ?, instructions = ?,
        dietary_tags = ?, image_url = ?, prep_time = ? WHERE id = ?
      `).run(
        args.title || existing.title,
        args.ingredients || existing.ingredients,
        args.instructions || existing.instructions,
        args.dietary_tags ? JSON.stringify(args.dietary_tags) : existing.dietary_tags,
        args.image_url !== undefined ? args.image_url : existing.image_url,
        args.prep_time !== undefined ? args.prep_time : existing.prep_time,
        args.id
      );
      return db.prepare('SELECT * FROM recipes WHERE id = ?').get(args.id);
    },

    deleteRecipe: (_, { id }) => {
      // TODO(session-16): No authorization check — any user can delete any recipe.
      // Same issue as updateRecipe. Needs auth middleware + ownership verification.
      const result = db.prepare('DELETE FROM recipes WHERE id = ?').run(id);
      return result.changes > 0;
    },

    shareRecipe: (_, { recipeId }) => {
      const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipeId);
      if (!recipe) return null;

      const existing = db.prepare('SELECT * FROM recipe_shares WHERE recipe_id = ?').get(recipeId);
      if (existing) return existing;

      const code = Math.random().toString(36).substring(2, 10);
      db.prepare('INSERT INTO recipe_shares (recipe_id, code) VALUES (?, ?)').run(recipeId, code);
      return db.prepare('SELECT * FROM recipe_shares WHERE recipe_id = ?').get(recipeId);
    },

    addIngredients: (_, { recipeId, ingredients }) => {
      const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipeId);
      if (!recipe) return null;

      const stmt = db.prepare(
        'INSERT INTO recipe_ingredients (recipe_id, name, amount, unit, category) VALUES (?, ?, ?, ?, ?)'
      );
      for (const ing of ingredients) {
        stmt.run(recipeId, ing.name, ing.amount, ing.unit, ing.category || 'other');
      }
      return db.prepare('SELECT * FROM recipe_ingredients WHERE recipe_id = ?').all(recipeId);
    },

    createMealPlan: (_, { weekStart, name }) => {
      const result = db.prepare('INSERT INTO meal_plans (week_start, name) VALUES (?, ?)').run(weekStart, name);
      return db.prepare('SELECT * FROM meal_plans WHERE id = ?').get(result.lastInsertRowid);
    },

    addMealPlanItem: (_, { planId, recipeId, dayOfWeek, mealType }) => {
      const result = db.prepare(
        'INSERT INTO meal_plan_items (plan_id, recipe_id, day_of_week, meal_type) VALUES (?, ?, ?, ?)'
      ).run(planId, recipeId, dayOfWeek, mealType);
      return db.prepare('SELECT * FROM meal_plan_items WHERE id = ?').get(result.lastInsertRowid);
    },

    removeMealPlanItem: (_, { id }) => {
      const result = db.prepare('DELETE FROM meal_plan_items WHERE id = ?').run(id);
      return result.changes > 0;
    },
  },

  // Field resolvers
  Recipe: {
    // N+1 BUG: This fires once PER recipe in a list query.
    // 100 recipes = 100 separate SQL queries for ingredients.
    ingredientList: (recipe) => {
      return db.prepare('SELECT * FROM recipe_ingredients WHERE recipe_id = ?').all(recipe.id);
    },
    dietaryTags: (recipe) => {
      return JSON.parse(recipe.dietary_tags || '[]');
    },
    imageUrl: (recipe) => recipe.image_url,
    prepTime: (recipe) => recipe.prep_time,
    createdAt: (recipe) => recipe.created_at,
  },

  MealPlan: {
    weekStart: (plan) => plan.week_start,
    createdAt: (plan) => plan.created_at,
    items: (plan) => {
      return db.prepare(`
        SELECT mpi.*, r.title AS recipe_title
        FROM meal_plan_items mpi
        JOIN recipes r ON r.id = mpi.recipe_id
        WHERE mpi.plan_id = ?
        ORDER BY mpi.day_of_week, mpi.meal_type
      `).all(plan.id);
    },
    groceryList: (plan) => {
      return db.prepare(`
        SELECT
          ri.name,
          ri.unit,
          ri.category,
          SUM(ri.amount) AS totalAmount,
          COUNT(DISTINCT mpi.recipe_id) AS recipeCount,
          GROUP_CONCAT(DISTINCT r.title) AS fromRecipes
        FROM meal_plan_items mpi
        JOIN recipe_ingredients ri ON ri.recipe_id = mpi.recipe_id
        JOIN recipes r ON r.id = mpi.recipe_id
        WHERE mpi.plan_id = ?
        GROUP BY ri.name, ri.unit, ri.category
        ORDER BY ri.category, ri.name
      `).all(plan.id);
    },
  },

  MealPlanItem: {
    dayOfWeek: (item) => item.day_of_week,
    mealType: (item) => item.meal_type,
    recipe: (item) => {
      return db.prepare('SELECT * FROM recipes WHERE id = ?').get(item.recipe_id);
    },
  },

  ShareLink: {
    recipeId: (share) => share.recipe_id,
    createdAt: (share) => share.created_at,
  },
};

module.exports = { resolvers };
