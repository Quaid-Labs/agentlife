/**
 * Meal Plan Tests — introduced session 10
 *
 * Tests meal plan creation, item management, and the grocery list
 * aggregation feature with GROUP BY logic for ingredient consolidation.
 */

const {
  initializeDatabase, resetDatabase, closeDatabase, getDb, createTestRecipe,
} = require('./setup');
const { VALID_DAYS, MEAL_TYPES } = require('./helpers');

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
// Helpers for meal plan operations
// ---------------------------------------------------------------------------

function createMealPlan(weekStart, name) {
  const stmt = db.prepare('INSERT INTO meal_plans (week_start, name) VALUES (?, ?)');
  const result = stmt.run(weekStart, name);
  return { id: result.lastInsertRowid, week_start: weekStart, name };
}

function addMealPlanItem(planId, recipeId, dayOfWeek, mealType) {
  const stmt = db.prepare(
    'INSERT INTO meal_plan_items (plan_id, recipe_id, day_of_week, meal_type) VALUES (?, ?, ?, ?)'
  );
  const result = stmt.run(planId, recipeId, dayOfWeek, mealType);
  return { id: result.lastInsertRowid, plan_id: planId, recipe_id: recipeId, day_of_week: dayOfWeek, meal_type: mealType };
}

function removeMealPlanItem(itemId) {
  return db.prepare('DELETE FROM meal_plan_items WHERE id = ?').run(itemId);
}

function getMealPlanWithItems(planId) {
  const plan = db.prepare('SELECT * FROM meal_plans WHERE id = ?').get(planId);
  if (!plan) return null;
  const items = db.prepare(`
    SELECT mpi.*, r.title AS recipe_title
    FROM meal_plan_items mpi
    JOIN recipes r ON r.id = mpi.recipe_id
    WHERE mpi.plan_id = ?
    ORDER BY mpi.day_of_week, mpi.meal_type
  `).all(planId);
  return { ...plan, items };
}

function addIngredient(recipeId, name, amount, unit, category = 'other') {
  db.prepare(
    'INSERT INTO recipe_ingredients (recipe_id, name, amount, unit, category) VALUES (?, ?, ?, ?, ?)'
  ).run(recipeId, name, amount, unit, category);
}

/**
 * Generate a grocery list for a meal plan using GROUP BY aggregation.
 * This mirrors the app logic: group ingredients by name+unit, sum amounts,
 * count how many times each ingredient appears (times_needed).
 */
function generateGroceryList(planId) {
  return db.prepare(`
    SELECT
      ri.name,
      ri.unit,
      ri.category,
      SUM(ri.amount) AS total_amount,
      COUNT(*) AS times_needed
    FROM meal_plan_items mpi
    JOIN recipe_ingredients ri ON ri.recipe_id = mpi.recipe_id
    WHERE mpi.plan_id = ?
    GROUP BY ri.name, ri.unit
    ORDER BY ri.category, ri.name
  `).all(planId);
}

// ---------------------------------------------------------------------------
// Meal Plan CRUD
// ---------------------------------------------------------------------------

describe('Meal Plan Creation', () => {
  it('should create a meal plan with week_start and name', () => {
    const plan = createMealPlan('2026-02-16', 'Week of Feb 16');

    expect(plan.id).toBeDefined();
    expect(plan.week_start).toBe('2026-02-16');
    expect(plan.name).toBe('Week of Feb 16');

    // Verify persistence
    const row = db.prepare('SELECT * FROM meal_plans WHERE id = ?').get(plan.id);
    expect(row.week_start).toBe('2026-02-16');
    expect(row.created_at).toBeDefined();
  });

  it('should auto-set created_at on meal plans', () => {
    const plan = createMealPlan('2026-02-16', 'Timestamped Plan');
    const row = db.prepare('SELECT created_at FROM meal_plans WHERE id = ?').get(plan.id);
    expect(row.created_at).not.toBeNull();
  });

  it('should allow multiple meal plans', () => {
    createMealPlan('2026-02-16', 'Week 1');
    createMealPlan('2026-02-23', 'Week 2');
    createMealPlan('2026-03-02', 'Week 3');

    const plans = db.prepare('SELECT * FROM meal_plans').all();
    expect(plans).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// Meal Plan Items
// ---------------------------------------------------------------------------

describe('Meal Plan Items', () => {
  let plan;
  let recipe1, recipe2;

  beforeEach(() => {
    recipe1 = createTestRecipe({ title: 'Morning Oats' });
    recipe2 = createTestRecipe({ title: 'Grilled Salmon' });
    plan = createMealPlan('2026-02-16', 'Test Plan');
  });

  it('should add an item to a meal plan', () => {
    const item = addMealPlanItem(plan.id, recipe1.id, 'monday', 'breakfast');

    expect(item.id).toBeDefined();
    expect(item.plan_id).toBe(plan.id);
    expect(item.recipe_id).toBe(recipe1.id);
    expect(item.day_of_week).toBe('monday');
    expect(item.meal_type).toBe('breakfast');
  });

  it('should add multiple items for the same day', () => {
    addMealPlanItem(plan.id, recipe1.id, 'monday', 'breakfast');
    addMealPlanItem(plan.id, recipe2.id, 'monday', 'dinner');

    const items = db.prepare('SELECT * FROM meal_plan_items WHERE plan_id = ? AND day_of_week = ?')
      .all(plan.id, 'monday');
    expect(items).toHaveLength(2);
  });

  it('should add the same recipe on different days', () => {
    addMealPlanItem(plan.id, recipe1.id, 'monday', 'breakfast');
    addMealPlanItem(plan.id, recipe1.id, 'wednesday', 'breakfast');
    addMealPlanItem(plan.id, recipe1.id, 'friday', 'breakfast');

    const items = db.prepare('SELECT * FROM meal_plan_items WHERE plan_id = ? AND recipe_id = ?')
      .all(plan.id, recipe1.id);
    expect(items).toHaveLength(3);
  });

  it('should remove an item from a meal plan', () => {
    const item = addMealPlanItem(plan.id, recipe1.id, 'tuesday', 'lunch');
    const result = removeMealPlanItem(item.id);

    expect(result.changes).toBe(1);
    const remaining = db.prepare('SELECT * FROM meal_plan_items WHERE plan_id = ?').all(plan.id);
    expect(remaining).toHaveLength(0);
  });

  it('should return changes=0 when removing non-existent item', () => {
    const result = removeMealPlanItem(99999);
    expect(result.changes).toBe(0);
  });

  it('should get a plan with all its items and recipe titles', () => {
    addMealPlanItem(plan.id, recipe1.id, 'monday', 'breakfast');
    addMealPlanItem(plan.id, recipe2.id, 'monday', 'dinner');
    addMealPlanItem(plan.id, recipe1.id, 'wednesday', 'breakfast');

    const fullPlan = getMealPlanWithItems(plan.id);

    expect(fullPlan.name).toBe('Test Plan');
    expect(fullPlan.items).toHaveLength(3);
    expect(fullPlan.items[0].recipe_title).toBe('Morning Oats');
    // Items should be sorted by day_of_week then meal_type
    expect(fullPlan.items[0].day_of_week).toBe('monday');
    expect(fullPlan.items[2].day_of_week).toBe('wednesday');
  });

  it('should return null for non-existent plan', () => {
    const result = getMealPlanWithItems(99999);
    expect(result).toBeNull();
  });

  it('should cascade delete items when plan is deleted', () => {
    addMealPlanItem(plan.id, recipe1.id, 'monday', 'breakfast');
    addMealPlanItem(plan.id, recipe2.id, 'tuesday', 'dinner');

    db.prepare('DELETE FROM meal_plans WHERE id = ?').run(plan.id);

    const items = db.prepare('SELECT * FROM meal_plan_items WHERE plan_id = ?').all(plan.id);
    expect(items).toHaveLength(0);
  });

  it('should cascade delete items when recipe is deleted', () => {
    addMealPlanItem(plan.id, recipe1.id, 'monday', 'breakfast');
    addMealPlanItem(plan.id, recipe2.id, 'tuesday', 'dinner');

    db.prepare('DELETE FROM recipes WHERE id = ?').run(recipe1.id);

    const items = db.prepare('SELECT * FROM meal_plan_items WHERE plan_id = ?').all(plan.id);
    expect(items).toHaveLength(1);
    expect(items[0].recipe_id).toBe(recipe2.id);
  });
});

// ---------------------------------------------------------------------------
// Grocery List Generation
// ---------------------------------------------------------------------------

describe('Grocery List Generation', () => {
  let plan;
  let pastaRecipe, saladRecipe, soupRecipe;

  beforeEach(() => {
    plan = createMealPlan('2026-02-16', 'Grocery Test Plan');

    pastaRecipe = createTestRecipe({ title: 'Pasta Primavera' });
    saladRecipe = createTestRecipe({ title: 'Green Salad' });
    soupRecipe = createTestRecipe({ title: 'Tomato Soup' });

    // Pasta ingredients
    addIngredient(pastaRecipe.id, 'pasta', 1, 'lbs', 'dry goods');
    addIngredient(pastaRecipe.id, 'olive oil', 2, 'tbsp', 'oils');
    addIngredient(pastaRecipe.id, 'garlic', 3, 'cloves', 'produce');
    addIngredient(pastaRecipe.id, 'parmesan', 0.5, 'cups', 'dairy');

    // Salad ingredients
    addIngredient(saladRecipe.id, 'romaine lettuce', 1, 'head', 'produce');
    addIngredient(saladRecipe.id, 'olive oil', 3, 'tbsp', 'oils');
    addIngredient(saladRecipe.id, 'lemon', 1, 'whole', 'produce');

    // Soup ingredients
    addIngredient(soupRecipe.id, 'tomatoes', 6, 'whole', 'produce');
    addIngredient(soupRecipe.id, 'garlic', 2, 'cloves', 'produce');
    addIngredient(soupRecipe.id, 'olive oil', 1, 'tbsp', 'oils');
    addIngredient(soupRecipe.id, 'cream', 0.25, 'cups', 'dairy');
  });

  it('should aggregate ingredients by name and unit', () => {
    addMealPlanItem(plan.id, pastaRecipe.id, 'monday', 'dinner');
    addMealPlanItem(plan.id, saladRecipe.id, 'monday', 'lunch');
    addMealPlanItem(plan.id, soupRecipe.id, 'tuesday', 'dinner');

    const grocery = generateGroceryList(plan.id);

    // Olive oil appears in all 3 recipes — should aggregate to 6 tbsp
    const oliveOil = grocery.find(g => g.name === 'olive oil');
    expect(oliveOil).toBeDefined();
    expect(oliveOil.total_amount).toBe(6); // 2 + 3 + 1
    expect(oliveOil.unit).toBe('tbsp');
    expect(oliveOil.times_needed).toBe(3);
  });

  it('should correctly count times_needed for each ingredient', () => {
    addMealPlanItem(plan.id, pastaRecipe.id, 'monday', 'dinner');
    addMealPlanItem(plan.id, soupRecipe.id, 'tuesday', 'dinner');

    const grocery = generateGroceryList(plan.id);

    // Garlic appears in pasta (3 cloves) and soup (2 cloves)
    const garlic = grocery.find(g => g.name === 'garlic');
    expect(garlic.total_amount).toBe(5); // 3 + 2
    expect(garlic.times_needed).toBe(2);

    // Pasta only appears once
    const pasta = grocery.find(g => g.name === 'pasta');
    expect(pasta.total_amount).toBe(1);
    expect(pasta.times_needed).toBe(1);
  });

  it('should aggregate same recipe used multiple days', () => {
    // Pasta on monday AND wednesday
    addMealPlanItem(plan.id, pastaRecipe.id, 'monday', 'dinner');
    addMealPlanItem(plan.id, pastaRecipe.id, 'wednesday', 'dinner');

    const grocery = generateGroceryList(plan.id);

    const pasta = grocery.find(g => g.name === 'pasta');
    expect(pasta.total_amount).toBe(2); // 1 + 1
    expect(pasta.times_needed).toBe(2);

    const garlic = grocery.find(g => g.name === 'garlic');
    expect(garlic.total_amount).toBe(6); // 3 + 3
    expect(garlic.times_needed).toBe(2);
  });

  it('should return empty array for empty meal plan', () => {
    const grocery = generateGroceryList(plan.id);
    expect(grocery).toHaveLength(0);
  });

  it('should return empty array for meal plan with no recipe ingredients', () => {
    const bareRecipe = createTestRecipe({ title: 'No Ingredients Defined' });
    // Recipe exists but has no entries in recipe_ingredients
    addMealPlanItem(plan.id, bareRecipe.id, 'monday', 'dinner');

    const grocery = generateGroceryList(plan.id);
    expect(grocery).toHaveLength(0);
  });

  it('should sort grocery list by category then name', () => {
    addMealPlanItem(plan.id, pastaRecipe.id, 'monday', 'dinner');
    addMealPlanItem(plan.id, saladRecipe.id, 'tuesday', 'lunch');
    addMealPlanItem(plan.id, soupRecipe.id, 'wednesday', 'dinner');

    const grocery = generateGroceryList(plan.id);

    // Verify categories are grouped (sorted alphabetically)
    const categories = grocery.map(g => g.category);
    for (let i = 1; i < categories.length; i++) {
      expect(categories[i] >= categories[i - 1]).toBe(true);
    }
  });

  it('should not mix ingredients with same name but different units', () => {
    // Add a second olive oil entry with a different unit
    addIngredient(pastaRecipe.id, 'olive oil', 0.25, 'cups', 'oils');
    addMealPlanItem(plan.id, pastaRecipe.id, 'monday', 'dinner');

    const grocery = generateGroceryList(plan.id);
    const oliveOils = grocery.filter(g => g.name === 'olive oil');

    // Should be two separate entries: 2 tbsp and 0.25 cups
    expect(oliveOils).toHaveLength(2);
    const tbspEntry = oliveOils.find(o => o.unit === 'tbsp');
    const cupsEntry = oliveOils.find(o => o.unit === 'cups');
    expect(tbspEntry.total_amount).toBe(2);
    expect(cupsEntry.total_amount).toBe(0.25);
  });
});

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

describe('Meal Plan Validation', () => {
  it('should enforce foreign key on plan_id', () => {
    const recipe = createTestRecipe({ title: 'Orphan Recipe' });

    expect(() => {
      addMealPlanItem(99999, recipe.id, 'monday', 'dinner');
    }).toThrow(/FOREIGN KEY/);
  });

  it('should enforce foreign key on recipe_id', () => {
    const plan = createMealPlan('2026-02-16', 'FK Test Plan');

    expect(() => {
      addMealPlanItem(plan.id, 99999, 'monday', 'dinner');
    }).toThrow(/FOREIGN KEY/);
  });
});
