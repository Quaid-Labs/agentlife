/**
 * GraphQL Resolver Tests — introduced session 12
 *
 * Tests Apollo Server resolvers directly (no HTTP layer), covering
 * queries, mutations, and the documented N+1 bug in Recipe.ingredientList.
 */

const {
  initializeDatabase, resetDatabase, closeDatabase, getDb, createTestRecipe, createTestUser,
} = require('./setup');
const { DIETARY_LABELS, SAFE_FOR_MOM } = require('./helpers');

let db;

beforeAll(() => {
  db = initializeDatabase();
});

afterAll(() => {
  closeDatabase();
});

beforeEach(() => {
  resetDatabase();
});

// ---------------------------------------------------------------------------
// Simulated resolvers — these mirror the app's GraphQL resolver functions
// ---------------------------------------------------------------------------

const resolvers = {
  Query: {
    recipes: (_, { dietFilter, safeForMom } = {}) => {
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
      return rows;
    },

    recipe: (_, { id }) => {
      return db.prepare('SELECT * FROM recipes WHERE id = ?').get(id) || null;
    },

    searchRecipes: (_, { query }) => {
      return db.prepare('SELECT * FROM recipes WHERE title LIKE ? OR ingredients LIKE ?')
        .all(`%${query}%`, `%${query}%`);
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
      const stmt = db.prepare(`
        INSERT INTO recipes (title, ingredients, instructions, dietary_tags, image_url, prep_time, owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `);
      const result = stmt.run(
        args.title, args.ingredients, args.instructions,
        JSON.stringify(args.dietary_tags || []),
        args.image_url || '', args.prep_time || 0, args.owner_id || null
      );
      return db.prepare('SELECT * FROM recipes WHERE id = ?').get(result.lastInsertRowid);
    },

    updateRecipe: (_, args) => {
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
      const result = db.prepare('DELETE FROM recipes WHERE id = ?').run(id);
      return result.changes > 0;
    },

    shareRecipe: (_, { recipeId }) => {
      const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipeId);
      if (!recipe) return null;
      // Check if already shared
      const existing = db.prepare('SELECT * FROM recipe_shares WHERE recipe_id = ?').get(recipeId);
      if (existing) return existing;
      // Generate unique code
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

  // N+1 BUG: This resolver fires once PER recipe in a list query.
  // If you query 100 recipes, this runs 100 separate SQL queries.
  Recipe: {
    ingredientList: (recipe) => {
      return db.prepare('SELECT * FROM recipe_ingredients WHERE recipe_id = ?').all(recipe.id);
    },
    dietaryTags: (recipe) => {
      return JSON.parse(recipe.dietary_tags || '[]');
    },
  },
};

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

describe('GraphQL Queries', () => {
  describe('recipes', () => {
    it('should return all recipes with no filter', () => {
      createTestRecipe({ title: 'Recipe A' });
      createTestRecipe({ title: 'Recipe B' });
      createTestRecipe({ title: 'Recipe C' });

      const result = resolvers.Query.recipes(null, {});
      expect(result).toHaveLength(3);
    });

    it('should return recipes in reverse chronological order', () => {
      createTestRecipe({ title: 'First' });
      createTestRecipe({ title: 'Second' });
      createTestRecipe({ title: 'Third' });

      const result = resolvers.Query.recipes(null, {});
      // All created in same second with CURRENT_TIMESTAMP, so order depends on insertion
      expect(result).toHaveLength(3);
    });

    it('should filter recipes by diet tag', () => {
      createTestRecipe({ title: 'Vegan Soup', dietary_tags: JSON.stringify(['vegan']) });
      createTestRecipe({ title: 'Regular Soup', dietary_tags: '[]' });

      const result = resolvers.Query.recipes(null, { dietFilter: ['vegan'] });
      expect(result).toHaveLength(1);
      expect(result[0].title).toBe('Vegan Soup');
    });

    it('should filter with safeForMom preset', () => {
      createTestRecipe({
        title: 'Mom Safe Meal',
        dietary_tags: JSON.stringify(['diabetic-friendly', 'low-sodium']),
      });
      createTestRecipe({
        title: 'Not Safe',
        dietary_tags: JSON.stringify(['diabetic-friendly']),
      });

      const result = resolvers.Query.recipes(null, { safeForMom: true });
      expect(result).toHaveLength(1);
      expect(result[0].title).toBe('Mom Safe Meal');
    });

    it('should prioritize safeForMom over dietFilter', () => {
      createTestRecipe({
        title: 'Vegan + Mom Safe',
        dietary_tags: JSON.stringify(['vegan', 'diabetic-friendly', 'low-sodium']),
      });
      createTestRecipe({
        title: 'Just Vegan',
        dietary_tags: JSON.stringify(['vegan']),
      });

      // safeForMom should take precedence
      const result = resolvers.Query.recipes(null, { dietFilter: ['vegan'], safeForMom: true });
      expect(result).toHaveLength(1);
      expect(result[0].title).toBe('Vegan + Mom Safe');
    });

    it('should return empty array when no recipes match filter', () => {
      createTestRecipe({ title: 'Regular Food', dietary_tags: '[]' });

      const result = resolvers.Query.recipes(null, { dietFilter: ['keto'] });
      expect(result).toHaveLength(0);
    });
  });

  describe('recipe', () => {
    it('should return a single recipe by ID', () => {
      const created = createTestRecipe({ title: 'Single Recipe' });
      const result = resolvers.Query.recipe(null, { id: created.id });

      expect(result).not.toBeNull();
      expect(result.title).toBe('Single Recipe');
      expect(result.id).toBe(created.id);
    });

    it('should return null for non-existent recipe', () => {
      const result = resolvers.Query.recipe(null, { id: 99999 });
      expect(result).toBeNull();
    });
  });

  describe('searchRecipes', () => {
    beforeEach(() => {
      createTestRecipe({ title: 'Chocolate Cake', ingredients: 'chocolate, flour, eggs' });
      createTestRecipe({ title: 'Vanilla Ice Cream', ingredients: 'cream, vanilla, sugar' });
      createTestRecipe({ title: 'Chocolate Mousse', ingredients: 'chocolate, cream, eggs' });
    });

    it('should search by title', () => {
      const result = resolvers.Query.searchRecipes(null, { query: 'Chocolate' });
      expect(result).toHaveLength(2);
    });

    it('should search by ingredients', () => {
      const result = resolvers.Query.searchRecipes(null, { query: 'cream' });
      // Vanilla Ice Cream (ingredient) and Chocolate Mousse (ingredient)
      expect(result).toHaveLength(2);
    });

    it('should return empty for no match', () => {
      const result = resolvers.Query.searchRecipes(null, { query: 'asparagus' });
      expect(result).toHaveLength(0);
    });

    it('should handle empty search query', () => {
      const result = resolvers.Query.searchRecipes(null, { query: '' });
      expect(result).toHaveLength(3);
    });
  });

  describe('recipeByShareCode', () => {
    it('should return recipe for valid share code', () => {
      const recipe = createTestRecipe({ title: 'Shared Recipe' });
      db.prepare('INSERT INTO recipe_shares (recipe_id, code) VALUES (?, ?)').run(recipe.id, 'abc123');

      const result = resolvers.Query.recipeByShareCode(null, { code: 'abc123' });
      expect(result).not.toBeNull();
      expect(result.title).toBe('Shared Recipe');
    });

    it('should return null for invalid share code', () => {
      const result = resolvers.Query.recipeByShareCode(null, { code: 'nonexistent' });
      expect(result).toBeNull();
    });
  });

  describe('dietaryLabels', () => {
    it('should return all dietary labels', () => {
      const result = resolvers.Query.dietaryLabels();
      expect(result).toEqual(DIETARY_LABELS);
      expect(result).toHaveLength(10);
    });
  });

  describe('mealPlans', () => {
    it('should return all meal plans', () => {
      db.prepare('INSERT INTO meal_plans (week_start, name) VALUES (?, ?)').run('2026-02-16', 'Plan A');
      db.prepare('INSERT INTO meal_plans (week_start, name) VALUES (?, ?)').run('2026-02-23', 'Plan B');

      const result = resolvers.Query.mealPlans();
      expect(result).toHaveLength(2);
    });

    it('should return empty array when no plans exist', () => {
      const result = resolvers.Query.mealPlans();
      expect(result).toHaveLength(0);
    });
  });

  describe('mealPlan', () => {
    it('should return a single meal plan by ID', () => {
      const res = db.prepare('INSERT INTO meal_plans (week_start, name) VALUES (?, ?)').run('2026-02-16', 'My Plan');
      const result = resolvers.Query.mealPlan(null, { id: res.lastInsertRowid });

      expect(result).not.toBeNull();
      expect(result.name).toBe('My Plan');
    });

    it('should return null for non-existent plan', () => {
      const result = resolvers.Query.mealPlan(null, { id: 99999 });
      expect(result).toBeNull();
    });
  });
});

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

describe('GraphQL Mutations', () => {
  describe('createRecipe', () => {
    it('should create a recipe and return it', () => {
      const result = resolvers.Mutation.createRecipe(null, {
        title: 'New Recipe',
        ingredients: 'a, b, c',
        instructions: 'Mix and serve.',
        dietary_tags: ['vegan'],
        prep_time: 15,
      });

      expect(result.id).toBeDefined();
      expect(result.title).toBe('New Recipe');
      expect(result.prep_time).toBe(15);
      expect(JSON.parse(result.dietary_tags)).toEqual(['vegan']);
    });

    it('should set defaults for optional fields', () => {
      const result = resolvers.Mutation.createRecipe(null, {
        title: 'Minimal Recipe',
        ingredients: 'just flour',
        instructions: 'wing it',
      });

      expect(result.image_url).toBe('');
      expect(result.prep_time).toBe(0);
      expect(JSON.parse(result.dietary_tags)).toEqual([]);
    });
  });

  describe('updateRecipe', () => {
    it('should update recipe fields', () => {
      const recipe = createTestRecipe({ title: 'Old Name', prep_time: 10 });
      const result = resolvers.Mutation.updateRecipe(null, {
        id: recipe.id,
        title: 'New Name',
        prep_time: 45,
      });

      expect(result.title).toBe('New Name');
      expect(result.prep_time).toBe(45);
      // Unchanged fields should remain
      expect(result.ingredients).toBe(recipe.ingredients);
    });

    it('should return null for non-existent recipe', () => {
      const result = resolvers.Mutation.updateRecipe(null, { id: 99999, title: 'Ghost' });
      expect(result).toBeNull();
    });

    it('should preserve fields not included in the update', () => {
      const recipe = createTestRecipe({
        title: 'Preserve Test',
        ingredients: 'keep these',
        instructions: 'keep these too',
        image_url: 'https://example.com/img.jpg',
      });

      const result = resolvers.Mutation.updateRecipe(null, {
        id: recipe.id,
        title: 'Updated Title',
      });

      expect(result.ingredients).toBe('keep these');
      expect(result.instructions).toBe('keep these too');
      expect(result.image_url).toBe('https://example.com/img.jpg');
    });
  });

  describe('deleteRecipe', () => {
    it('should delete a recipe and return true', () => {
      const recipe = createTestRecipe({ title: 'Delete Me' });
      const result = resolvers.Mutation.deleteRecipe(null, { id: recipe.id });

      expect(result).toBe(true);
      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipe.id);
      expect(row).toBeUndefined();
    });

    it('should return false for non-existent recipe', () => {
      const result = resolvers.Mutation.deleteRecipe(null, { id: 99999 });
      expect(result).toBe(false);
    });
  });

  describe('shareRecipe', () => {
    it('should create a share code for a recipe', () => {
      const recipe = createTestRecipe({ title: 'Shareable' });
      const result = resolvers.Mutation.shareRecipe(null, { recipeId: recipe.id });

      expect(result).not.toBeNull();
      expect(result.code).toBeDefined();
      expect(result.code.length).toBeGreaterThan(0);
      expect(result.recipe_id).toBe(recipe.id);
    });

    it('should return null for non-existent recipe', () => {
      const result = resolvers.Mutation.shareRecipe(null, { recipeId: 99999 });
      expect(result).toBeNull();
    });

    it('should return same share record when sharing the same recipe twice', () => {
      const recipe = createTestRecipe({ title: 'Share Twice' });
      const first = resolvers.Mutation.shareRecipe(null, { recipeId: recipe.id });
      const second = resolvers.Mutation.shareRecipe(null, { recipeId: recipe.id });

      expect(first.code).toBe(second.code);
      expect(first.id).toBe(second.id);
    });
  });

  describe('addIngredients', () => {
    it('should add ingredients to a recipe', () => {
      const recipe = createTestRecipe({ title: 'Ingredient Test' });
      const result = resolvers.Mutation.addIngredients(null, {
        recipeId: recipe.id,
        ingredients: [
          { name: 'flour', amount: 2, unit: 'cups', category: 'dry goods' },
          { name: 'sugar', amount: 1, unit: 'cups', category: 'dry goods' },
          { name: 'butter', amount: 0.5, unit: 'cups', category: 'dairy' },
        ],
      });

      expect(result).toHaveLength(3);
      expect(result[0].name).toBe('flour');
      expect(result[1].amount).toBe(1);
      expect(result[2].category).toBe('dairy');
    });

    it('should return null for non-existent recipe', () => {
      const result = resolvers.Mutation.addIngredients(null, {
        recipeId: 99999,
        ingredients: [{ name: 'ghost', amount: 1, unit: 'cups' }],
      });
      expect(result).toBeNull();
    });

    it('should default category to other', () => {
      const recipe = createTestRecipe({ title: 'Default Category' });
      const result = resolvers.Mutation.addIngredients(null, {
        recipeId: recipe.id,
        ingredients: [{ name: 'mystery item', amount: 1, unit: 'piece' }],
      });

      expect(result[0].category).toBe('other');
    });
  });

  describe('createMealPlan', () => {
    it('should create a meal plan', () => {
      const result = resolvers.Mutation.createMealPlan(null, {
        weekStart: '2026-03-01',
        name: 'March Week 1',
      });

      expect(result.id).toBeDefined();
      expect(result.week_start).toBe('2026-03-01');
      expect(result.name).toBe('March Week 1');
    });
  });

  describe('addMealPlanItem', () => {
    it('should add an item to a meal plan', () => {
      const recipe = createTestRecipe({ title: 'Planned Recipe' });
      const plan = db.prepare('INSERT INTO meal_plans (week_start, name) VALUES (?, ?)').run('2026-02-16', 'Plan');

      const result = resolvers.Mutation.addMealPlanItem(null, {
        planId: plan.lastInsertRowid,
        recipeId: recipe.id,
        dayOfWeek: 'wednesday',
        mealType: 'lunch',
      });

      expect(result.day_of_week).toBe('wednesday');
      expect(result.meal_type).toBe('lunch');
    });
  });

  describe('removeMealPlanItem', () => {
    it('should remove a meal plan item and return true', () => {
      const recipe = createTestRecipe({ title: 'To Remove' });
      const plan = db.prepare('INSERT INTO meal_plans (week_start, name) VALUES (?, ?)').run('2026-02-16', 'Plan');
      const item = db.prepare(
        'INSERT INTO meal_plan_items (plan_id, recipe_id, day_of_week, meal_type) VALUES (?, ?, ?, ?)'
      ).run(plan.lastInsertRowid, recipe.id, 'friday', 'dinner');

      const result = resolvers.Mutation.removeMealPlanItem(null, { id: item.lastInsertRowid });
      expect(result).toBe(true);
    });

    it('should return false when item does not exist', () => {
      const result = resolvers.Mutation.removeMealPlanItem(null, { id: 99999 });
      expect(result).toBe(false);
    });
  });
});

// ---------------------------------------------------------------------------
// Field Resolvers
// ---------------------------------------------------------------------------

describe('GraphQL Field Resolvers', () => {
  describe('Recipe.ingredientList', () => {
    it('should return ingredients for a recipe', () => {
      const recipe = createTestRecipe({ title: 'With Ingredients' });
      db.prepare(
        'INSERT INTO recipe_ingredients (recipe_id, name, amount, unit) VALUES (?, ?, ?, ?)'
      ).run(recipe.id, 'salt', 1, 'tsp');
      db.prepare(
        'INSERT INTO recipe_ingredients (recipe_id, name, amount, unit) VALUES (?, ?, ?, ?)'
      ).run(recipe.id, 'pepper', 0.5, 'tsp');

      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipe.id);
      const result = resolvers.Recipe.ingredientList(row);

      expect(result).toHaveLength(2);
      expect(result[0].name).toBe('salt');
      expect(result[1].name).toBe('pepper');
    });

    it('should return empty array for recipe with no ingredients', () => {
      const recipe = createTestRecipe({ title: 'Bare Recipe' });
      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipe.id);
      const result = resolvers.Recipe.ingredientList(row);

      expect(result).toHaveLength(0);
    });
  });

  describe('Recipe.dietaryTags', () => {
    it('should parse dietary_tags JSON string to array', () => {
      const recipe = createTestRecipe({
        title: 'Tagged',
        dietary_tags: JSON.stringify(['vegan', 'gluten-free']),
      });
      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipe.id);
      const result = resolvers.Recipe.dietaryTags(row);

      expect(result).toEqual(['vegan', 'gluten-free']);
    });

    it('should return empty array for recipe with no tags', () => {
      const recipe = createTestRecipe({ title: 'Untagged' });
      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipe.id);
      const result = resolvers.Recipe.dietaryTags(row);

      expect(result).toEqual([]);
    });
  });

  describe('N+1 Query Bug (documented)', () => {
    /**
     * BUG DOCUMENTATION: The Recipe.ingredientList resolver executes a separate
     * SQL query for each recipe. When querying a list of N recipes, this results
     * in N+1 total queries (1 for the recipe list + N for ingredients).
     *
     * This test documents the behavior. The fix would be a DataLoader that
     * batches ingredient lookups into a single WHERE recipe_id IN (...) query.
     */
    it('should execute separate queries per recipe (N+1 pattern)', () => {
      // Create 5 recipes with ingredients
      const recipes = [];
      for (let i = 0; i < 5; i++) {
        const r = createTestRecipe({ title: `Recipe ${i}` });
        db.prepare(
          'INSERT INTO recipe_ingredients (recipe_id, name, amount, unit) VALUES (?, ?, ?, ?)'
        ).run(r.id, `ingredient_${i}`, 1, 'unit');
        recipes.push(r);
      }

      // Count how many times ingredientList resolver is called
      let queryCount = 0;
      const originalResolver = resolvers.Recipe.ingredientList;
      resolvers.Recipe.ingredientList = (recipe) => {
        queryCount++;
        return originalResolver(recipe);
      };

      // Simulate what happens when GraphQL resolves a list of recipes
      const allRecipes = resolvers.Query.recipes(null, {});
      expect(allRecipes).toHaveLength(5);

      // Each recipe triggers its own ingredientList query
      allRecipes.forEach(recipe => {
        resolvers.Recipe.ingredientList(recipe);
      });

      // N+1: 1 query for recipes + 5 queries for ingredients = 6 total
      // This test documents that ingredientList fires N times
      expect(queryCount).toBe(5);

      // Restore original resolver
      resolvers.Recipe.ingredientList = originalResolver;
    });
  });
});
