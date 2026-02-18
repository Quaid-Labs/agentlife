/**
 * Database seeder — populates recipes, ingredients, and a sample meal plan.
 *
 * Reads sample-recipes.json (20 recipes across 4 dietary categories),
 * parses ingredient text into structured recipe_ingredients rows, and
 * creates a sample weekly meal plan for quick demo/dev use.
 *
 * Introduced: session 10 (seed data + structured ingredients)
 *
 * Usage:
 *   node seeds/seed.js          # run directly
 *   npm run seed                # via package.json script
 */

const path = require('path');
const { createConnection } = require('../config/database');
const recipes = require('./sample-recipes.json');

/** Dietary tag combination that is safe for Linda (mom) */
const SAFE_FOR_MOM = ['diabetic-friendly', 'low-sodium'];

/**
 * Parse a comma-separated ingredient string into structured objects.
 *
 * Attempts to extract amount, unit, and name from patterns like
 * "2 cups flour" or "1/2 tsp salt". Falls back to the full text
 * as the name with amount 1 and unit "whole" when parsing fails.
 */
function parseIngredients(text) {
  return text.split(',').map((item) => {
    const trimmed = item.trim();
    // Match: "2 cups flour", "1.5 lbs chicken", "1/2 tsp salt", "3 cloves garlic minced"
    const match = trimmed.match(/^([\d./]+)\s+(\w+)\s+(.+)$/);
    if (match) {
      return {
        amount: parseFloat(eval(match[1])) || 1, // handles fractions like 1/2
        unit: match[2],
        name: match[3],
      };
    }
    // No amount/unit detected — treat as 1 whole item
    return { amount: 1, unit: 'whole', name: trimmed };
  });
}

/**
 * Guess a grocery category from an ingredient name.
 */
function categorize(name) {
  const lower = name.toLowerCase();
  if (/chicken|beef|pork|turkey|salmon|cod|shrimp|sausage/.test(lower)) return 'protein';
  if (/milk|cream|cheese|butter|yogurt|parmesan|mozzarella/.test(lower)) return 'dairy';
  if (/flour|sugar|rice|pasta|spaghetti|linguine|bread|bun|tortilla|quinoa|lentil|panko/.test(lower)) return 'grains';
  if (/oil|vinegar|soy sauce|sesame|hoisin|sriracha|hot sauce|bbq|marinara/.test(lower)) return 'condiments';
  if (/salt|pepper|cumin|paprika|oregano|basil|thyme|rosemary|turmeric|garam|chili|cayenne|garlic powder|onion powder|italian seasoning/.test(lower)) return 'spices';
  if (/can |canned|tomato paste|broth|coconut milk|diced tomatoes|crushed tomatoes/.test(lower)) return 'canned goods';
  return 'produce';
}

function seed() {
  const db = createConnection();
  console.log('Seeding database...');

  // Clear existing data (order respects foreign keys)
  db.exec('DELETE FROM recipe_ingredients');
  db.exec('DELETE FROM meal_plan_items');
  db.exec('DELETE FROM meal_plans');
  db.exec('DELETE FROM recipe_shares');
  db.exec('DELETE FROM recipes');

  const insertRecipe = db.prepare(`
    INSERT INTO recipes (title, ingredients, instructions, dietary_tags, image_url, prep_time)
    VALUES (@title, @ingredients, @instructions, @dietary_tags, @image_url, @prep_time)
  `);

  const insertIngredient = db.prepare(`
    INSERT INTO recipe_ingredients (recipe_id, name, amount, unit, category)
    VALUES (@recipe_id, @name, @amount, @unit, @category)
  `);

  const insertPlan = db.prepare(`
    INSERT INTO meal_plans (week_start, name) VALUES (@week_start, @name)
  `);

  const insertPlanItem = db.prepare(`
    INSERT INTO meal_plan_items (plan_id, recipe_id, day_of_week, meal_type)
    VALUES (@plan_id, @recipe_id, @day_of_week, @meal_type)
  `);

  let totalIngredients = 0;
  const recipeIds = [];

  // --- Seed recipes + ingredients in a single transaction ---
  db.transaction(() => {
    for (const recipe of recipes) {
      const result = insertRecipe.run({
        title: recipe.title,
        ingredients: recipe.ingredients,
        instructions: recipe.instructions,
        dietary_tags: JSON.stringify(recipe.dietary_tags),
        image_url: recipe.image_url,
        prep_time: recipe.prep_time,
      });

      const recipeId = Number(result.lastInsertRowid);
      recipeIds.push({ id: recipeId, title: recipe.title, tags: recipe.dietary_tags });

      const ingredients = parseIngredients(recipe.ingredients);
      for (const ing of ingredients) {
        insertIngredient.run({
          recipe_id: recipeId,
          name: ing.name,
          amount: ing.amount,
          unit: ing.unit,
          category: categorize(ing.name),
        });
        totalIngredients++;
      }
    }
  })();

  // --- Create a sample meal plan (Mon-Fri) ---
  const safeRecipes = recipeIds.filter((r) =>
    SAFE_FOR_MOM.every((tag) => r.tags.includes(tag))
  );
  const vegRecipes = recipeIds.filter((r) => r.tags.includes('vegetarian'));

  const planResult = insertPlan.run({
    week_start: '2026-02-16',
    name: 'Family Meal Plan',
  });
  const planId = Number(planResult.lastInsertRowid);

  const weekDays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'];
  let planItemCount = 0;

  db.transaction(() => {
    for (let i = 0; i < weekDays.length; i++) {
      // Dinner: alternate between safe-for-mom and vegetarian-friendly
      const pool = i % 2 === 0 ? safeRecipes : vegRecipes;
      const recipe = pool[i % pool.length];
      insertPlanItem.run({
        plan_id: planId,
        recipe_id: recipe.id,
        day_of_week: weekDays[i],
        meal_type: 'dinner',
      });
      planItemCount++;
    }
  })();

  console.log(`Seeded ${recipes.length} recipes`);
  console.log(`Seeded ${totalIngredients} ingredient entries`);
  console.log(`Seeded 1 meal plan with ${planItemCount} items`);
  console.log('Done.');

  db.close();
}

if (require.main === module) {
  seed();
}

module.exports = { seed, parseIngredients, categorize };
